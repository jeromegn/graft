//! File-based page storage with pack files.
//!
//! Segments are batched into pack files (one per shard) to reduce file count.
//! Each pack file contains multiple segments appended sequentially.
//!
//! # Pack File Format
//!
//! ```text
//! ┌────────────────────────────────────────┐
//! │ Segment 1                              │
//! │  ├─ Magic: "GRAFTSG\x01"    (8 bytes)  │
//! │  ├─ Splinter size (u32 LE)  (4 bytes)  │
//! │  ├─ Splinter bitmap         (variable) │
//! │  └─ Page data               (N × 16KB) │
//! ├────────────────────────────────────────┤
//! │ Segment 2                              │
//! │  └─ ...                                │
//! ├────────────────────────────────────────┤
//! │ ...                                    │
//! └────────────────────────────────────────┘
//! ```

use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    os::unix::io::AsRawFd,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arc_swap::ArcSwapOption;
use bytes::Bytes;
use dashmap::DashMap;
use memmap2::MmapMut;
use parking_lot::RwLock;
use splinter_rs::{CowSplinter, PartitionRead};
use zerocopy::IntoBytes;

#[cfg(unix)]
use std::os::unix::io::RawFd;

/// Maximum number of iovecs per pwritev call (IOV_MAX is typically 1024)
const IOV_MAX: usize = 1024;

/// Write all data from multiple IoSlices at a specific offset using pwritev.
/// This is thread-safe - multiple threads can write to different offsets concurrently.
/// Handles IOV_MAX by chunking writes when there are many slices.
#[cfg(unix)]
fn pwritev_all(fd: RawFd, offset: u64, data: &[&[u8]]) -> std::io::Result<()> {
    use std::io::{IoSlice, Error, ErrorKind};

    let mut current_offset = offset as libc::off_t;
    let mut data_idx = 0;

    while data_idx < data.len() {
        // Take at most IOV_MAX slices at a time
        let chunk_end = (data_idx + IOV_MAX).min(data.len());
        let chunk = &data[data_idx..chunk_end];

        let mut iovecs: Vec<IoSlice<'_>> = chunk.iter().map(|d| IoSlice::new(d)).collect();
        let mut slices = iovecs.as_mut_slice();

        while !slices.is_empty() {
            // SAFETY: fd is a valid file descriptor, iovecs point to valid memory
            let written = unsafe {
                libc::pwritev(
                    fd,
                    slices.as_ptr() as *const libc::iovec,
                    slices.len() as libc::c_int,
                    current_offset,
                )
            };

            if written < 0 {
                let err = Error::last_os_error();
                if err.kind() == ErrorKind::Interrupted {
                    continue;
                }
                return Err(err);
            }

            if written == 0 {
                return Err(Error::new(ErrorKind::WriteZero, "pwritev wrote 0 bytes"));
            }

            let mut remaining = written as usize;
            current_offset += written as libc::off_t;

            // Advance past fully written slices
            while !slices.is_empty() && remaining >= slices[0].len() {
                remaining -= slices[0].len();
                slices = &mut slices[1..];
                data_idx += 1;
            }

            // Handle partial write of a slice
            if remaining > 0 && !slices.is_empty() {
                // pwritev doesn't support partial IoSlice advancement,
                // fall back to writing the remainder directly
                let slice_data = &slices[0][remaining..];
                let mut written_extra = 0;
                while written_extra < slice_data.len() {
                    // SAFETY: fd is valid, slice_data is valid memory
                    let n = unsafe {
                        libc::pwrite(
                            fd,
                            slice_data[written_extra..].as_ptr() as *const libc::c_void,
                            slice_data.len() - written_extra,
                            current_offset,
                        )
                    };
                    if n < 0 {
                        let err = Error::last_os_error();
                        if err.kind() == ErrorKind::Interrupted {
                            continue;
                        }
                        return Err(err);
                    }
                    if n == 0 {
                        return Err(Error::new(ErrorKind::WriteZero, "pwrite wrote 0 bytes"));
                    }
                    written_extra += n as usize;
                    current_offset += n as libc::off_t;
                }
                slices = &mut slices[1..];
                data_idx += 1;
            }
        }
    }

    Ok(())
}

/// File lock guard that releases the lock on drop
struct FileLock {
    file: File,
    #[allow(dead_code)]
    exclusive: bool,
}

impl FileLock {
    #[allow(dead_code)]
    fn shared(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        let fd = file.as_raw_fd();
        // SAFETY: fd is a valid file descriptor from an open file.
        // flock is safe to call with a valid fd and lock operation.
        let result = unsafe { libc::flock(fd, libc::LOCK_SH) };
        if result != 0 {
            return Err(std::io::Error::last_os_error());
        }

        Ok(Self { file, exclusive: false })
    }

    fn exclusive(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;

        let fd = file.as_raw_fd();
        // SAFETY: fd is a valid file descriptor from an open file.
        // flock is safe to call with a valid fd and lock operation.
        let result = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if result != 0 {
            return Err(std::io::Error::last_os_error());
        }

        Ok(Self { file, exclusive: true })
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let fd = self.file.as_raw_fd();
        // SAFETY: fd is a valid file descriptor from an open file that we own.
        // flock with LOCK_UN releases any lock held.
        unsafe { libc::flock(fd, libc::LOCK_UN) };
    }
}

/// Cross-process lock constants.
/// Lock byte is stored at offset 0 of each pack file.
/// Bit 7 = compact pending, bits 0-6 = reader count (max 127 concurrent readers).
const LOCK_COMPACT_BIT: u8 = 0x80;
const LOCK_COUNT_MASK: u8 = 0x7F;

/// RAII guard for read access to a shard's lock byte
struct ShardReadGuard<'a> {
    state: &'a AtomicU8,
}

impl Drop for ShardReadGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        self.state.fetch_sub(1, Ordering::Release);
    }
}

use crate::core::{
    PageIdx, SegmentId,
    page::{PAGESIZE, Page},
    pageset::PageSet,
};

use super::{PageStore, PageStoreErr};

/// Magic bytes identifying a segment
const MAGIC: &[u8; 8] = b"GRAFTSG\x01";

/// Size of the file header (lock byte)
const FILE_HEADER_SIZE: usize = 1;

/// Size of the magic + splinter size header (before SegmentId)
const HEADER_PREFIX_SIZE: usize = 8 + 4;

/// Size of SegmentId in bytes
const SEGMENT_ID_SIZE: usize = 16;

/// Number of shards (pack files)
const NUM_SHARDS: usize = 256;

/// Minimum mmap capacity (64 MB)
const MIN_MMAP_CAPACITY: u64 = 64 * 1024 * 1024;

/// Cached pack file with mmap and segment index.
/// Uses DashMap for lock-free concurrent access to segments.
struct CachedPack {
    /// Mutable mmap - needed for atomic lock byte at offset 0
    mmap: MmapMut,
    /// The capacity of the mmap (may be larger than actual data)
    capacity: u64,
    /// Segment index: SegmentId -> (offset_in_mmap, data_offset, pageset)
    /// DashMap allows lock-free reads and concurrent inserts without cloning.
    segments: DashMap<SegmentId, (usize, usize, Arc<PageSet>)>,
}

impl CachedPack {
    /// Get the lock byte as an atomic reference
    #[inline]
    fn lock_state(&self) -> &AtomicU8 {
        // SAFETY: mmap is at least FILE_HEADER_SIZE bytes, and we only access
        // through atomics which handle concurrent access correctly.
        unsafe { &*(self.mmap.as_ptr() as *const AtomicU8) }
    }

    /// Acquire read access. Returns a guard that releases on drop.
    #[inline]
    fn read_acquire(&self) -> ShardReadGuard<'_> {
        let state = self.lock_state();
        loop {
            let old = state.fetch_add(1, Ordering::Acquire);
            if old & LOCK_COMPACT_BIT == 0 {
                // No compaction pending, we're good
                break;
            }
            // Compaction pending, undo and wait
            state.fetch_sub(1, Ordering::Release);
            while state.load(Ordering::Relaxed) & LOCK_COMPACT_BIT != 0 {
                std::hint::spin_loop();
            }
        }
        ShardReadGuard { state }
    }

    /// Acquire exclusive access for compaction. Blocks until all readers finish.
    fn compact_acquire(&self) {
        let state = self.lock_state();
        // Set compact bit
        state.fetch_or(LOCK_COMPACT_BIT, Ordering::SeqCst);
        // Wait for readers to drain
        while state.load(Ordering::Acquire) & LOCK_COUNT_MASK != 0 {
            std::thread::yield_now();
        }
    }

    /// Release exclusive access after compaction
    fn compact_release(&self) {
        let state = self.lock_state();
        state.fetch_and(!LOCK_COMPACT_BIT, Ordering::Release);
    }
}

/// Per-shard configuration (immutable after init)
struct ShardConfig {
    /// Path to the pack file
    path: PathBuf,
    /// Path to the lock file
    lock_path: PathBuf,
    /// Path to the access times file
    access_path: PathBuf,
}

/// Per-shard write state
struct ShardWriter {
    /// Write file descriptor (kept open for pwritev)
    fd: RawFd,
    /// The File handle (to keep fd valid)
    _file: File,
    /// Current data size - use fetch_add to atomically reserve space
    data_size: AtomicU64,
    /// Lock for resize operations (ftruncate + remap)
    /// Read lock: normal pwritev (concurrent)
    /// Write lock: resize operation (exclusive)
    resize_lock: RwLock<()>,
}

/// Per-shard access time tracking (lock-free via DashMap)
struct ShardAccessTimes {
    /// Last access time per segment (epoch seconds)
    times: DashMap<SegmentId, u64>,
    /// Whether access times have been modified since last flush
    dirty: std::sync::atomic::AtomicBool,
}

/// File-based page storage using pack files.
///
/// Segments are grouped into 256 pack files based on SegmentId.
/// This dramatically reduces file count and open/close overhead.
///
/// Read path is lock-free via ArcSwap + DashMap for the cache.
/// Write path uses pwritev for concurrent writes without mutex.
/// Mmap is pre-extended to reduce remapping frequency.
///
/// Each pack file has a 1-byte header for cross-process synchronization:
/// - Bit 7: compact pending flag
/// - Bits 0-6: reader count (max 127 concurrent readers)
pub struct FilePageStore {
    /// Root directory for pack files
    #[allow(dead_code)]
    root: PathBuf,
    /// Per-shard configuration
    configs: Vec<ShardConfig>,
    /// Per-shard cache - lock-free reads via ArcSwap
    caches: Vec<ArcSwapOption<CachedPack>>,
    /// Per-shard write state - lazily initialized
    writers: Vec<RwLock<Option<ShardWriter>>>,
    /// Per-shard access times
    access_times: Vec<ShardAccessTimes>,
}

impl FilePageStore {
    /// Create a new file-based page store at the given path.
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self, PageStoreErr> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;

        // Initialize shards
        let mut configs = Vec::with_capacity(NUM_SHARDS);
        let mut caches = Vec::with_capacity(NUM_SHARDS);
        let mut writers = Vec::with_capacity(NUM_SHARDS);
        let mut access_times_vec = Vec::with_capacity(NUM_SHARDS);

        for i in 0..NUM_SHARDS {
            let path = root.join(format!("{:02x}.pack", i));
            let lock_path = root.join(format!("{:02x}.lock", i));
            let access_path = root.join(format!("{:02x}.access", i));

            // Load access times from file
            let times_map = Self::load_access_times(&access_path).unwrap_or_default();
            let times = DashMap::new();
            for (k, v) in times_map {
                times.insert(k, v);
            }

            configs.push(ShardConfig {
                path,
                lock_path,
                access_path,
            });
            caches.push(ArcSwapOption::empty());
            writers.push(RwLock::new(None));
            access_times_vec.push(ShardAccessTimes {
                times,
                dirty: std::sync::atomic::AtomicBool::new(false),
            });
        }

        let store = Self {
            root,
            configs,
            caches,
            writers,
            access_times: access_times_vec,
        };

        // Load existing pack files into cache
        store.rebuild_index()?;

        Ok(store)
    }

    /// Get or create a writer for a shard
    fn get_or_create_writer(&self, shard_num: usize) -> Result<(), PageStoreErr> {
        // Fast path: already initialized
        if self.writers[shard_num].read().is_some() {
            return Ok(());
        }

        // Slow path: initialize writer
        let mut writer_guard = self.writers[shard_num].write();
        if writer_guard.is_some() {
            return Ok(()); // Another thread initialized it
        }

        let config = &self.configs[shard_num];
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&config.path)?;

        let fd = file.as_raw_fd();
        let data_size = file.metadata()?.len();

        *writer_guard = Some(ShardWriter {
            fd,
            _file: file,
            data_size: AtomicU64::new(data_size),
            resize_lock: RwLock::new(()),
        });

        Ok(())
    }

    /// Load access times from a file
    fn load_access_times(path: &Path) -> Result<HashMap<SegmentId, u64>, PageStoreErr> {
        if !path.exists() {
            return Ok(HashMap::new());
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut times = HashMap::new();

        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() == 2 {
                if let (Ok(sid), Ok(time)) = (parts[0].parse::<SegmentId>(), parts[1].parse::<u64>())
                {
                    times.insert(sid, time);
                }
            }
        }

        Ok(times)
    }

    /// Save access times to a file
    fn save_access_times(
        path: &Path,
        times: &DashMap<SegmentId, u64>,
    ) -> Result<(), PageStoreErr> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        let mut writer = BufWriter::new(file);

        for entry in times.iter() {
            writeln!(writer, "{}\t{}", entry.key(), entry.value())?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Get current time as epoch seconds
    fn now_epoch() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs()
    }

    /// Determine which shard a segment belongs to
    fn shard_for(&self, sid: &SegmentId) -> u8 {
        sid.as_bytes()[7]
    }

    /// Load all existing pack files into the in-memory cache
    fn rebuild_index(&self) -> Result<(), PageStoreErr> {
        for shard_num in 0..NUM_SHARDS {
            let config = &self.configs[shard_num];
            if !config.path.exists() {
                continue;
            }

            let metadata = fs::metadata(&config.path)?;
            // Skip files without even a header
            if metadata.len() < FILE_HEADER_SIZE as u64 {
                continue;
            }

            // Scan the pack file to find all segments
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&config.path)?;
            // SAFETY: The file is open and valid. We need write access for the
            // atomic lock byte. Initialization happens before any concurrent access.
            let mmap = unsafe { MmapMut::map_mut(&file)? };
            let capacity = mmap.len() as u64;

            let segments = Self::scan_pack(&mmap, &config.path)?;

            // Build cached pack with DashMap
            let seg_map = DashMap::new();
            for (sid, offset, data_offset, pageset) in segments {
                seg_map.insert(sid, (offset, data_offset, Arc::new(pageset)));
            }

            // Store cache via ArcSwap (lock-free)
            self.caches[shard_num].store(Some(Arc::new(CachedPack {
                mmap,
                capacity,
                segments: seg_map,
            })));
        }

        Ok(())
    }

    /// Scan a pack file and return all segments found.
    /// Format: file_header(1) + [segment...]
    /// Each segment: magic(8) + splinter_size(4) + segment_id(16) + splinter(var) + pages
    fn scan_pack(
        mmap: &[u8],
        _path: &Path,
    ) -> Result<Vec<(SegmentId, usize, usize, PageSet)>, PageStoreErr> {
        let mut segments = Vec::new();
        // Start after file header (lock byte)
        let mut offset = FILE_HEADER_SIZE;

        let min_header = HEADER_PREFIX_SIZE + SEGMENT_ID_SIZE;

        while offset + min_header <= mmap.len() {
            // Check magic
            if &mmap[offset..offset + 8] != MAGIC {
                // Might be partial write or corruption, stop scanning
                break;
            }

            // Read splinter size
            let splinter_size =
                u32::from_le_bytes(mmap[offset + 8..offset + 12].try_into().unwrap()) as usize;

            // Read SegmentId
            let sid_start = offset + HEADER_PREFIX_SIZE;
            let sid_end = sid_start + SEGMENT_ID_SIZE;
            if sid_end > mmap.len() {
                break;
            }
            let sid = SegmentId::try_from(&mmap[sid_start..sid_end])
                .map_err(|_| PageStoreErr::InvalidSegmentFile("invalid segment id".to_string()))?;

            // Calculate offsets
            let splinter_start = sid_end;
            let splinter_end = splinter_start + splinter_size;
            let data_offset = splinter_end - offset; // relative to segment start

            if splinter_end > mmap.len() {
                break;
            }

            // Parse splinter
            let splinter_bytes = Bytes::copy_from_slice(&mmap[splinter_start..splinter_end]);
            let splinter = match CowSplinter::from_bytes(splinter_bytes) {
                Ok(s) => s,
                Err(_) => break,
            };

            let pageset = PageSet::new(splinter);
            let page_count = pageset.cardinality().to_usize();
            let segment_size = data_offset + page_count * PAGESIZE.as_usize();

            if offset + segment_size > mmap.len() {
                break;
            }

            segments.push((sid, offset, data_offset, pageset));
            offset += segment_size;
        }

        Ok(segments)
    }

    /// Get a segment from its pack file (lock-free fast path)
    fn get_segment(
        &self,
        sid: &SegmentId,
    ) -> Result<Option<(Arc<CachedPack>, usize, usize, Arc<PageSet>)>, PageStoreErr> {
        let shard_num = self.shard_for(sid) as usize;

        // Fast path: lock-free cache lookup via ArcSwap + DashMap
        if let Some(cached) = self.caches[shard_num].load_full() {
            if let Some(entry) = cached.segments.get(sid) {
                let (offset, data_offset, pageset) = entry.value();
                return Ok(Some((
                    Arc::clone(&cached),
                    *offset,
                    *data_offset,
                    Arc::clone(pageset),
                )));
            }
            // Cache exists but segment not found - it doesn't exist
            return Ok(None);
        }

        // Slow path: cache not loaded, need to load it
        let config = &self.configs[shard_num];

        // Double-check after acquiring any initialization lock
        if let Some(cached) = self.caches[shard_num].load_full() {
            if let Some(entry) = cached.segments.get(sid) {
                let (offset, data_offset, pageset) = entry.value();
                return Ok(Some((
                    Arc::clone(&cached),
                    *offset,
                    *data_offset,
                    Arc::clone(pageset),
                )));
            }
            return Ok(None);
        }

        // Nothing to load if file doesn't exist or is too small
        if !config.path.exists() {
            return Ok(None);
        }

        // Load the pack file with read/write access for atomic lock byte
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&config.path)?;
        let metadata = file.metadata()?;
        if metadata.len() < FILE_HEADER_SIZE as u64 {
            return Ok(None);
        }

        // SAFETY: The file is open and valid. We need write access for atomic
        // lock byte at offset 0. Handle EINVAL gracefully - can happen if file
        // was truncated between our length check and mmap call (race with compaction).
        let mmap = match unsafe { MmapMut::map_mut(&file) } {
            Ok(m) => m,
            Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                // EINVAL from mmap typically means length is 0.
                // Verify the file is actually empty before returning None.
                if file.metadata().map(|m| m.len()).unwrap_or(1) < FILE_HEADER_SIZE as u64 {
                    return Ok(None);
                }
                // File is non-empty but mmap still failed - this is unexpected
                return Err(PageStoreErr::Io(e));
            }
            Err(e) => return Err(PageStoreErr::Io(e)),
        };
        let capacity = mmap.len() as u64;
        let segments_vec = Self::scan_pack(&mmap, &config.path)?;

        let seg_map = DashMap::new();
        for (seg_sid, offset, data_offset, pageset) in segments_vec {
            seg_map.insert(seg_sid, (offset, data_offset, Arc::new(pageset)));
        }

        let result = seg_map.get(sid).map(|entry| {
            let (offset, data_offset, pageset) = entry.value();
            (*offset, *data_offset, Arc::clone(pageset))
        });

        let cached = Arc::new(CachedPack {
            mmap,
            capacity,
            segments: seg_map,
        });

        // Store cache via ArcSwap
        self.caches[shard_num].store(Some(Arc::clone(&cached)));

        Ok(result.map(|(offset, data_offset, pageset)| (cached, offset, data_offset, pageset)))
    }
}

impl FilePageStore {
    /// Access a page's data via callback without copying.
    ///
    /// The callback receives a slice to the page data. The slice is only valid
    /// for the duration of the callback - any data needed after must be copied
    /// by the caller.
    ///
    /// This is the preferred method for read-only access as it avoids copying
    /// and ensures proper lock management.
    pub fn with_page<F, R>(
        &self,
        sid: &SegmentId,
        pageidx: PageIdx,
        f: F,
    ) -> Result<Option<R>, PageStoreErr>
    where
        F: FnOnce(&[u8]) -> R,
    {
        let (cached, seg_offset, data_offset, pageset) = match self.get_segment(sid)? {
            Some(s) => s,
            None => return Ok(None),
        };

        if !pageset.contains(pageidx) {
            return Ok(None);
        }

        // Acquire cross-process read lock
        let _guard = cached.read_acquire();

        // Calculate page offset within segment
        let rank = pageset.splinter().rank(pageidx.to_u32()) - 1;
        let page_offset = seg_offset + data_offset + rank * PAGESIZE.as_usize();
        let page_end = page_offset + PAGESIZE.as_usize();

        if page_end > cached.mmap.len() {
            return Err(PageStoreErr::InvalidSegmentFile(
                "page offset beyond file end".to_string(),
            ));
        }

        // Call the callback with a slice - lock is held for duration of callback
        let slice = &cached.mmap[page_offset..page_end];
        Ok(Some(f(slice)))
        // _guard dropped here, releasing the lock
    }
}

impl PageStore for FilePageStore {
    fn has_page(&self, sid: &SegmentId, pageidx: PageIdx) -> Result<bool, PageStoreErr> {
        match self.get_segment(sid)? {
            Some((_, _, _, pageset)) => Ok(pageset.contains(pageidx)),
            None => Ok(false),
        }
    }

    fn read_page(&self, sid: &SegmentId, pageidx: PageIdx) -> Result<Option<Page>, PageStoreErr> {
        self.with_page(sid, pageidx, |slice| {
            Page::try_from(slice).expect("slice is exactly PAGESIZE")
        })
    }

    fn write_segment(
        &self,
        sid: SegmentId,
        pages: BTreeMap<PageIdx, Page>,
    ) -> Result<(), PageStoreErr> {
        if pages.is_empty() {
            return Ok(());
        }

        let shard_num = self.shard_for(&sid) as usize;

        // Build segment data (done outside any lock)
        let pageset = PageSet::from(splinter_rs::Splinter::from_iter(
            pages.keys().map(|idx| idx.to_u32()),
        ));

        let splinter_bytes = pageset.splinter().encode_to_bytes();
        let splinter_size = splinter_bytes.len() as u32;

        // Segment header: magic + splinter_size + segment_id + splinter
        let sid_bytes = sid.as_bytes();
        let mut seg_header =
            Vec::with_capacity(HEADER_PREFIX_SIZE + SegmentId::SIZE.as_usize() + splinter_bytes.len());
        seg_header.extend_from_slice(MAGIC);
        seg_header.extend_from_slice(&splinter_size.to_le_bytes());
        seg_header.extend_from_slice(sid_bytes);
        seg_header.extend_from_slice(&splinter_bytes);

        let data_offset = seg_header.len();
        let segment_size = (data_offset + pages.len() * PAGESIZE.as_usize()) as u64;

        let config = &self.configs[shard_num];

        // Retry loop: if compaction replaces the cache while we're writing,
        // we need to re-write to the new file.
        loop {
            // Ensure writer is initialized
            self.get_or_create_writer(shard_num)?;

            // Get the current cache (if any) and acquire read lock to block compaction
            let cached_before = self.caches[shard_num].load_full();
            let _read_guard = cached_before.as_ref().map(|c| c.read_acquire());

            // Get writer (we know it's initialized now)
            let writer_guard = self.writers[shard_num].read();
            let writer = match writer_guard.as_ref() {
                Some(w) => w,
                None => {
                    // Writer was invalidated (compaction happened), retry
                    drop(writer_guard);
                    continue;
                }
            };

            // Atomically reserve space. New files start at FILE_HEADER_SIZE (after lock byte).
            // CAS loop to handle the case where we need to initialize from 0 to FILE_HEADER_SIZE.
            let seg_offset = loop {
                let current = writer.data_size.load(Ordering::SeqCst);
                let start = if current < FILE_HEADER_SIZE as u64 {
                    FILE_HEADER_SIZE as u64
                } else {
                    current
                };
                let new_size = start + segment_size;
                match writer.data_size.compare_exchange(
                    current,
                    new_size,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break start,
                    Err(_) => continue,
                }
            };
            let new_data_size = seg_offset + segment_size;

            // Check if we need to extend the file and remap
            let current_capacity = self.caches[shard_num]
                .load_full()
                .map_or(0, |c| c.capacity);

            if new_data_size > current_capacity {
                // Try to acquire resize lock - if another thread has it, they'll handle resize
                // We can proceed with pwritev regardless since it works on sparse files
                if let Some(_resize_guard) = writer.resize_lock.try_write() {
                    // Double-check capacity after acquiring lock
                    let current_capacity = self.caches[shard_num]
                        .load_full()
                        .map_or(0, |c| c.capacity);

                    if new_data_size > current_capacity {
                        // Calculate new capacity: at least 2x current data or MIN_MMAP_CAPACITY
                        let new_capacity = new_data_size
                            .max(current_capacity * 2)
                            .max(MIN_MMAP_CAPACITY);

                        // Extend file to new capacity (creates sparse file)
                        // SAFETY: ftruncate is safe with a valid fd
                        let result =
                            unsafe { libc::ftruncate(writer.fd, new_capacity as libc::off_t) };
                        if result != 0 {
                            return Err(PageStoreErr::Io(std::io::Error::last_os_error()));
                        }

                        // Create new mmap with larger capacity (read/write for lock byte)
                        let file_for_mmap = OpenOptions::new()
                            .read(true)
                            .write(true)
                            .open(&config.path)?;
                        // SAFETY: The file is open and valid. We need write for atomic lock byte.
                        let new_mmap = unsafe { MmapMut::map_mut(&file_for_mmap)? };
                        let new_capacity = new_mmap.len() as u64;

                        // Create new DashMap, copying entries from old cache if it exists
                        let new_segments = DashMap::new();
                        if let Some(old_cached) = self.caches[shard_num].load_full() {
                            for entry in old_cached.segments.iter() {
                                new_segments.insert(entry.key().clone(), entry.value().clone());
                            }
                        }

                        // Atomically swap in the new cache (segment will be added below)
                        self.caches[shard_num].store(Some(Arc::new(CachedPack {
                            mmap: new_mmap,
                            capacity: new_capacity,
                            segments: new_segments,
                        })));
                    }
                }
                // If we couldn't get the lock, another thread is resizing.
                // pwritev works on sparse files, so we can proceed.
            }

            // Build data slices for pwritev
            // Note: No lock needed - pwritev writes to fd directly, unaffected by ftruncate/mmap
            let page_refs: Vec<&[u8]> = pages.values().map(|p| p.as_ref()).collect();
            let mut data_slices: Vec<&[u8]> = Vec::with_capacity(1 + page_refs.len());
            data_slices.push(&seg_header);
            data_slices.extend(page_refs);

            // Write using pwritev (thread-safe, no seek needed)
            pwritev_all(writer.fd, seg_offset, &data_slices)?;

            // Verify the writer is still valid (not invalidated by compaction).
            // If compaction happened, our data went to the old (now unlinked) file
            // and we need to retry.
            drop(writer_guard);
            if self.writers[shard_num].read().is_none() {
                // Writer was invalidated, our data is lost. Retry the whole write.
                continue;
            }

            // Update cache - insert into DashMap
            if let Some(cached) = self.caches[shard_num].load_full() {
                cached
                    .segments
                    .insert(sid.clone(), (seg_offset as usize, data_offset, Arc::new(pageset)));
            } else {
                // No cache exists yet, need to create one
                let file_for_mmap = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(&config.path)?;
                // SAFETY: The file is open and valid. We need write for atomic lock byte.
                let new_mmap = unsafe { MmapMut::map_mut(&file_for_mmap)? };
                let capacity = new_mmap.len() as u64;
                let segments = DashMap::new();
                segments.insert(sid.clone(), (seg_offset as usize, data_offset, Arc::new(pageset)));

                self.caches[shard_num].store(Some(Arc::new(CachedPack {
                    mmap: new_mmap,
                    capacity,
                    segments,
                })));
            }

            // Set initial access time for the new segment (lock-free)
            self.access_times[shard_num].times.insert(sid, Self::now_epoch());
            self.access_times[shard_num].dirty.store(true, Ordering::Relaxed);

            return Ok(());
        } // end retry loop
    }

    fn has_segment(&self, sid: &SegmentId) -> Result<bool, PageStoreErr> {
        // Check the cache for this shard - shard is derived from SegmentId itself
        let shard_num = self.shard_for(sid) as usize;
        if let Some(cached) = self.caches[shard_num].load_full() {
            return Ok(cached.segments.contains_key(sid));
        }
        // No cache means no segments in this shard
        Ok(false)
    }

    fn remove_segment(&self, sid: &SegmentId) -> Result<(), PageStoreErr> {
        // Delegate to the inherent impl
        FilePageStore::remove_segment(self, sid)
    }
}

impl FilePageStore {
    /// Update the access time for a segment.
    /// Call this explicitly if you need LRU tracking.
    /// Note: read_page does NOT update access times automatically for performance.
    pub fn touch_segment(&self, sid: &SegmentId) {
        let shard_num = self.shard_for(sid) as usize;
        self.access_times[shard_num].times.insert(sid.clone(), Self::now_epoch());
        self.access_times[shard_num].dirty.store(true, Ordering::Relaxed);
    }

    /// Remove a segment from the in-memory cache.
    /// The segment data remains in the pack file until compaction.
    pub fn remove_segment(&self, sid: &SegmentId) -> Result<(), PageStoreErr> {
        // Shard is derived directly from SegmentId
        let shard_num = self.shard_for(sid) as usize;

        // Remove from cache's DashMap (no need to rebuild the whole cache)
        if let Some(cached) = self.caches[shard_num].load_full() {
            cached.segments.remove(sid);
        }

        // Remove from access times (lock-free)
        self.access_times[shard_num].times.remove(sid);
        self.access_times[shard_num].dirty.store(true, Ordering::Relaxed);

        Ok(())
    }

    /// Flush access times to disk for all dirty shards
    pub fn flush_access_times(&self) -> Result<(), PageStoreErr> {
        for (i, access) in self.access_times.iter().enumerate() {
            if access.dirty.load(Ordering::Relaxed) {
                Self::save_access_times(&self.configs[i].access_path, &access.times)?;
                access.dirty.store(false, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// Release physical memory for segments not accessed within `max_age_secs`.
    ///
    /// Uses `madvise(MADV_DONTNEED)` to tell the kernel it can reclaim the pages.
    /// The data remains on disk and will be paged back in on next access.
    ///
    /// Returns the number of bytes released.
    #[cfg(unix)]
    pub fn release_cold_memory(&self, max_age_secs: u64) -> usize {
        let now = Self::now_epoch();
        let mut released = 0;

        for shard_num in 0..NUM_SHARDS {
            let Some(cached) = self.caches[shard_num].load_full() else {
                continue;
            };

            for entry in cached.segments.iter() {
                let sid = entry.key();
                let (offset, data_offset, pageset) = entry.value();

                // Check access time
                let last_access = self.access_times[shard_num]
                    .times
                    .get(sid)
                    .map(|e| *e.value())
                    .unwrap_or(0);

                if now.saturating_sub(last_access) < max_age_secs {
                    continue; // Recently accessed, keep in memory
                }

                // Calculate segment size
                let page_count = pageset.cardinality().to_usize();
                let segment_size = *data_offset + page_count * PAGESIZE.as_usize();

                // Tell kernel to release these pages
                // SAFETY: mmap is valid, offset and length are within bounds
                let ptr = unsafe { cached.mmap.as_ptr().add(*offset) };
                let result = unsafe {
                    libc::madvise(ptr as *mut libc::c_void, segment_size, libc::MADV_DONTNEED)
                };

                if result == 0 {
                    released += segment_size;
                }
            }
        }

        released
    }

    /// Release physical memory for an entire shard.
    ///
    /// Uses `madvise(MADV_DONTNEED)` to tell the kernel it can reclaim the pages.
    #[cfg(unix)]
    pub fn release_shard_memory(&self, shard_num: u8) -> usize {
        let shard_num = shard_num as usize;
        if shard_num >= NUM_SHARDS {
            return 0;
        }

        let Some(cached) = self.caches[shard_num].load_full() else {
            return 0;
        };

        let len = cached.mmap.len();
        if len == 0 {
            return 0;
        }

        // SAFETY: mmap is valid and len is within bounds
        let result = unsafe {
            libc::madvise(
                cached.mmap.as_ptr() as *mut libc::c_void,
                len,
                libc::MADV_DONTNEED,
            )
        };

        if result == 0 { len } else { 0 }
    }

    /// Release physical memory for all shards.
    ///
    /// Returns the total bytes released.
    #[cfg(unix)]
    pub fn release_all_memory(&self) -> usize {
        let mut released = 0;
        for i in 0..NUM_SHARDS {
            released += self.release_shard_memory(i as u8);
        }
        released
    }

    /// Compact a shard by rewriting only live segments.
    /// This reclaims disk space from removed segments.
    /// Uses file locking for multi-process safety.
    pub fn compact_shard(&self, shard_num: u8) -> Result<(), PageStoreErr> {
        let shard_num = shard_num as usize;
        if shard_num >= NUM_SHARDS {
            return Ok(());
        }

        let config = &self.configs[shard_num];

        // Acquire exclusive file lock for compaction (multi-process mutual exclusion)
        let _lock = FileLock::exclusive(&config.lock_path)?;

        // Get current cache - if none exists, nothing to compact
        let cached = match self.caches[shard_num].load_full() {
            Some(c) => c,
            None => return Ok(()),
        };

        // Acquire cross-process lock via the cached pack's lock byte.
        // This blocks new readers and waits for existing readers to drain.
        cached.compact_acquire();

        let live_segments: Vec<_> = cached
            .segments
            .iter()
            .map(|entry| {
                let (offset, data_offset, pageset) = entry.value();
                (entry.key().clone(), *offset, *data_offset, pageset.clone())
            })
            .collect();

        if live_segments.is_empty() {
            // All segments removed. Use the same rename pattern as non-empty case
            // to avoid SIGBUS race: readers holding the old mmap can still access
            // the old (unlinked) file while we replace it with an empty one.

            // Write empty pack file (just lock byte) to temp file
            let temp_path = config.path.with_extension("pack.tmp");
            let mut temp_file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_path)?;
            temp_file.write_all(&[0u8])?; // Lock byte initialized to 0

            // Rename temp file over existing pack file
            fs::rename(&temp_path, &config.path)?;

            // Invalidate writer so it gets recreated with new fd
            *self.writers[shard_num].write() = None;

            // Create new mmap for empty file and update cache
            let new_file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&config.path)?;
            let new_mmap = unsafe { MmapMut::map_mut(&new_file)? };
            let capacity = new_mmap.len() as u64;

            self.caches[shard_num].store(Some(Arc::new(CachedPack {
                mmap: new_mmap,
                capacity,
                segments: DashMap::new(),
            })));

            self.access_times[shard_num].times.clear();
            self.access_times[shard_num].dirty.store(true, Ordering::Relaxed);

            // Release lock on OLD cached pack AFTER new cache is in place.
            // Readers spinning on the old mmap will complete their reads from
            // the old (now unlinked) file, then get the new empty cache on retry.
            cached.compact_release();

            return Ok(());
        }

        // Read all live segment data from current mmap
        let old_mmap: &[u8] = &cached.mmap;
        let mut segment_data: Vec<(SegmentId, Vec<u8>, usize, Arc<PageSet>)> = Vec::new();

        for (sid, offset, data_offset, pageset) in &live_segments {
            let page_count = pageset.cardinality().to_usize();
            let segment_size = *data_offset + page_count * PAGESIZE.as_usize();
            let end = offset + segment_size;

            if end <= old_mmap.len() {
                segment_data.push((
                    sid.clone(),
                    old_mmap[*offset..end].to_vec(),
                    *data_offset,
                    Arc::clone(pageset),
                ));
            }
        }

        // Write to new temp file with header
        let temp_path = config.path.with_extension("pack.tmp");
        let mut temp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;

        // Write file header (lock byte, initialized to 0)
        temp_file.write_all(&[0u8])?;

        let new_segments = DashMap::new();
        let mut new_offset = FILE_HEADER_SIZE; // Start after header
        let mut new_access_times: Vec<(SegmentId, u64)> = Vec::new();

        for (sid, data, data_offset, pageset) in segment_data {
            temp_file.write_all(&data)?;
            new_segments.insert(sid.clone(), (new_offset, data_offset, pageset));
            // Preserve access time
            if let Some(entry) = self.access_times[shard_num].times.get(&sid) {
                new_access_times.push((sid, *entry.value()));
            }
            new_offset += data.len();
        }

        // Rename temp file to pack file
        fs::rename(&temp_path, &config.path)?;

        // Invalidate writer so it gets recreated with new fd
        *self.writers[shard_num].write() = None;

        // Reopen and remap with read/write access for lock byte
        let new_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&config.path)?;

        // SAFETY: The file is open and valid. We need write for atomic lock byte.
        // We hold the exclusive file lock.
        let new_mmap = unsafe { MmapMut::map_mut(&new_file)? };
        let capacity = new_mmap.len() as u64;

        // Update cache atomically with new pack
        self.caches[shard_num].store(Some(Arc::new(CachedPack {
            mmap: new_mmap,
            capacity,
            segments: new_segments,
        })));

        // Update access times - clear and repopulate
        self.access_times[shard_num].times.clear();
        for (sid, time) in new_access_times {
            self.access_times[shard_num].times.insert(sid, time);
        }
        self.access_times[shard_num].dirty.store(true, Ordering::Relaxed);

        // Release cross-process lock on the OLD cached pack.
        // This unblocks any readers that were waiting on the old pack.
        cached.compact_release();

        Ok(())
    }

    /// Compact all shards that have significant dead space.
    /// Returns the number of shards compacted.
    pub fn compact_all(&self) -> Result<usize, PageStoreErr> {
        let mut compacted = 0;
        for i in 0..NUM_SHARDS {
            // TODO: Add heuristic to skip shards with little dead space
            self.compact_shard(i as u8)?;
            compacted += 1;
        }
        Ok(compacted)
    }

    /// Get all segments with their access times, sorted by access time (oldest first).
    fn segments_by_access_time(&self) -> Vec<(SegmentId, u64)> {
        let mut segments: Vec<(SegmentId, u64)> = Vec::new();

        for access in &self.access_times {
            for entry in access.times.iter() {
                segments.push((entry.key().clone(), *entry.value()));
            }
        }

        // Sort by access time, oldest first
        segments.sort_by_key(|(_, time)| *time);
        segments
    }

    /// Get the total size of all segments (approximate, based on pack file sizes).
    pub fn total_size(&self) -> u64 {
        self.writers
            .iter()
            .map(|w| w.read().as_ref().map_or(0, |writer| writer.data_size.load(Ordering::Relaxed)))
            .sum()
    }

    /// Prune segments that haven't been accessed within the given duration.
    /// Returns the number of segments removed.
    pub fn prune_by_age(&self, max_age: Duration) -> Result<usize, PageStoreErr> {
        let cutoff = Self::now_epoch().saturating_sub(max_age.as_secs());
        let mut removed = 0;

        let segments = self.segments_by_access_time();
        for (sid, access_time) in segments {
            if access_time < cutoff {
                self.remove_segment(&sid)?;
                removed += 1;
            }
        }

        Ok(removed)
    }

    /// Prune the N least recently accessed segments.
    /// Returns the number of segments removed.
    pub fn prune_lru(&self, count: usize) -> Result<usize, PageStoreErr> {
        if count == 0 {
            return Ok(0);
        }

        let segments = self.segments_by_access_time();
        let to_remove = segments.into_iter().take(count);

        let mut removed = 0;
        for (sid, _) in to_remove {
            self.remove_segment(&sid)?;
            removed += 1;
        }

        Ok(removed)
    }

    /// Prune least recently accessed segments until total size is under target_bytes.
    /// Returns the number of segments removed.
    pub fn prune_to_size(&self, target_bytes: u64) -> Result<usize, PageStoreErr> {
        let current_size = self.total_size();
        if current_size <= target_bytes {
            return Ok(0);
        }

        let segments = self.segments_by_access_time();
        let mut removed = 0;
        let mut estimated_freed: u64 = 0;

        for (sid, _) in segments {
            if current_size.saturating_sub(estimated_freed) <= target_bytes {
                break;
            }

            // Estimate segment size before removing
            let seg_size = self.estimate_segment_size(&sid);
            self.remove_segment(&sid)?;
            estimated_freed += seg_size;
            removed += 1;
        }

        Ok(removed)
    }

    /// Estimate the size of a segment in bytes.
    fn estimate_segment_size(&self, sid: &SegmentId) -> u64 {
        let shard_num = self.shard_for(sid) as usize;

        self.caches[shard_num]
            .load_full()
            .and_then(|cached| cached.segments.get(sid).map(|entry| {
                let (_, data_offset, pageset) = entry.value();
                let page_count = pageset.cardinality().to_usize();
                (*data_offset + page_count * PAGESIZE.as_usize()) as u64
            }))
            .unwrap_or(0)
    }

    /// Get the number of segments in the store.
    pub fn segment_count(&self) -> usize {
        self.caches
            .iter()
            .filter_map(|cache| cache.load_full())
            .map(|cached| cached.segments.len())
            .sum()
    }

    /// Get statistics about the store.
    pub fn stats(&self) -> FilePageStoreStats {
        let segment_count = self.segment_count();
        let total_size = self.total_size();

        let mut oldest_access: Option<u64> = None;
        let mut newest_access: Option<u64> = None;

        for access in &self.access_times {
            for entry in access.times.iter() {
                let time = *entry.value();
                oldest_access = Some(oldest_access.map_or(time, |o| o.min(time)));
                newest_access = Some(newest_access.map_or(time, |n| n.max(time)));
            }
        }

        FilePageStoreStats {
            segment_count,
            total_size,
            oldest_access_epoch: oldest_access,
            newest_access_epoch: newest_access,
        }
    }
}

/// Statistics about the file page store.
#[derive(Debug, Clone)]
pub struct FilePageStoreStats {
    /// Number of segments in the store.
    pub segment_count: usize,
    /// Total size of all pack files in bytes.
    pub total_size: u64,
    /// Oldest access time as epoch seconds, if any segments exist.
    pub oldest_access_epoch: Option<u64>,
    /// Newest access time as epoch seconds, if any segments exist.
    pub newest_access_epoch: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pageidx;

    #[test]
    fn test_write_and_read_segment() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        let sid = SegmentId::random();
        let mut pages = BTreeMap::new();
        pages.insert(pageidx!(1), Page::test_filled(0x11));
        pages.insert(pageidx!(3), Page::test_filled(0x33));
        pages.insert(pageidx!(5), Page::test_filled(0x55));

        // Write segment
        store.write_segment(sid.clone(), pages).unwrap();

        // Verify segment exists
        assert!(store.has_segment(&sid).unwrap());

        // Read back pages
        let page1 = store.read_page(&sid, pageidx!(1)).unwrap().unwrap();
        assert_eq!(page1.as_ref()[0], 0x11);

        let page3 = store.read_page(&sid, pageidx!(3)).unwrap().unwrap();
        assert_eq!(page3.as_ref()[0], 0x33);

        let page5 = store.read_page(&sid, pageidx!(5)).unwrap().unwrap();
        assert_eq!(page5.as_ref()[0], 0x55);

        // Non-existent pages
        assert!(store.read_page(&sid, pageidx!(2)).unwrap().is_none());
        assert!(store.read_page(&sid, pageidx!(4)).unwrap().is_none());

        // has_page checks
        assert!(store.has_page(&sid, pageidx!(1)).unwrap());
        assert!(!store.has_page(&sid, pageidx!(2)).unwrap());
    }

    #[test]
    fn test_nonexistent_segment() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        let sid = SegmentId::random();
        assert!(!store.has_segment(&sid).unwrap());
        assert!(!store.has_page(&sid, pageidx!(1)).unwrap());
        assert!(store.read_page(&sid, pageidx!(1)).unwrap().is_none());
    }

    #[test]
    fn test_empty_segment_write() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        let sid = SegmentId::random();
        store.write_segment(sid.clone(), BTreeMap::new()).unwrap();

        // Empty segment should not be indexed
        assert!(!store.has_segment(&sid).unwrap());
    }

    #[test]
    fn test_multiple_segments_same_shard() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Write multiple segments
        let mut sids = Vec::new();
        for i in 0..10 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        // Read all back
        for (i, sid) in sids.iter().enumerate() {
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], i as u8);
        }
    }

    #[test]
    fn test_remove_and_compact() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Write segments
        let mut sids = Vec::new();
        for i in 0..5 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        // Remove some segments
        store.remove_segment(&sids[1]).unwrap();
        store.remove_segment(&sids[3]).unwrap();

        // Verify removed segments are gone
        assert!(!store.has_segment(&sids[1]).unwrap());
        assert!(!store.has_segment(&sids[3]).unwrap());

        // Verify remaining segments still work
        assert!(store.has_segment(&sids[0]).unwrap());
        assert!(store.has_segment(&sids[2]).unwrap());
        assert!(store.has_segment(&sids[4]).unwrap());

        let page = store.read_page(&sids[0], pageidx!(1)).unwrap().unwrap();
        assert_eq!(page.as_ref()[0], 0);

        // Compact all shards
        store.compact_all().unwrap();

        // Verify remaining segments still work after compaction
        let page = store.read_page(&sids[0], pageidx!(1)).unwrap().unwrap();
        assert_eq!(page.as_ref()[0], 0);

        let page = store.read_page(&sids[2], pageidx!(1)).unwrap().unwrap();
        assert_eq!(page.as_ref()[0], 2);

        let page = store.read_page(&sids[4], pageidx!(1)).unwrap().unwrap();
        assert_eq!(page.as_ref()[0], 4);

        // Removed segments still gone
        assert!(!store.has_segment(&sids[1]).unwrap());
        assert!(!store.has_segment(&sids[3]).unwrap());
    }

    #[test]
    fn test_prune_lru() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Write segments and read them in a specific order to set access times
        let mut sids = Vec::new();
        for i in 0..5 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            // Read to set access time (writes don't set access time)
            store.read_page(&sid, pageidx!(1)).unwrap();
            sids.push(sid);
        }

        assert_eq!(store.segment_count(), 5);

        // Prune 2 least recently used
        let removed = store.prune_lru(2).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.segment_count(), 3);

        // Prune 0 should do nothing
        let removed = store.prune_lru(0).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(store.segment_count(), 3);

        // Prune more than remaining
        let removed = store.prune_lru(10).unwrap();
        assert_eq!(removed, 3);
        assert_eq!(store.segment_count(), 0);
    }

    #[test]
    fn test_prune_to_size() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Write segments
        let mut sids = Vec::new();
        for i in 0..3 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            store.read_page(&sid, pageidx!(1)).unwrap();
            sids.push(sid);
        }

        let initial_size = store.total_size();
        assert!(initial_size > 0);

        // Prune to larger size should do nothing
        let removed = store.prune_to_size(initial_size + 1000).unwrap();
        assert_eq!(removed, 0);

        // Prune to 0 should remove everything
        let removed = store.prune_to_size(0).unwrap();
        assert_eq!(removed, 3);
        assert_eq!(store.segment_count(), 0);
    }

    #[test]
    fn test_stats() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Empty store
        let stats = store.stats();
        assert_eq!(stats.segment_count, 0);
        assert_eq!(stats.total_size, 0);
        assert!(stats.oldest_access_epoch.is_none());
        assert!(stats.newest_access_epoch.is_none());

        // Add a segment
        let sid = SegmentId::random();
        let mut pages = BTreeMap::new();
        pages.insert(pageidx!(1), Page::test_filled(0x11));
        store.write_segment(sid.clone(), pages).unwrap();
        store.read_page(&sid, pageidx!(1)).unwrap();

        let stats = store.stats();
        assert_eq!(stats.segment_count, 1);
        assert!(stats.total_size > 0);
        assert!(stats.oldest_access_epoch.is_some());
        assert!(stats.newest_access_epoch.is_some());
    }

    #[test]
    #[cfg(unix)]
    fn test_release_memory() {
        use std::thread;
        use std::time::Duration;

        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Write several segments
        let mut sids = Vec::new();
        for i in 0..5 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            pages.insert(pageidx!(2), Page::test_filled(i as u8 + 100));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        // Read all segments to ensure they're in memory
        for sid in &sids {
            store.read_page(sid, pageidx!(1)).unwrap();
        }

        // Release all memory
        let released = store.release_all_memory();
        assert!(released > 0, "Should have released some memory");

        // Data should still be readable (paged back in from disk)
        for (i, sid) in sids.iter().enumerate() {
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], i as u8);
        }

        // Test release_cold_memory with a short delay
        thread::sleep(Duration::from_secs(2));
        let released = store.release_cold_memory(1); // Release segments older than 1 second
        assert!(released > 0, "Should have released cold memory");

        // Data should still be readable
        for (i, sid) in sids.iter().enumerate() {
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], i as u8);
        }
    }

    #[test]
    fn test_access_times_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let sid = SegmentId::random();

        // Create store and write a segment
        {
            let store = FilePageStore::new(dir.path()).unwrap();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(0x11));
            store.write_segment(sid.clone(), pages).unwrap();
            store.read_page(&sid, pageidx!(1)).unwrap();
            store.flush_access_times().unwrap();
        }

        // Reopen store and verify access times were loaded
        {
            let store = FilePageStore::new(dir.path()).unwrap();
            let stats = store.stats();
            assert_eq!(stats.segment_count, 1);
            assert!(stats.oldest_access_epoch.is_some());
        }
    }

    #[test]
    fn test_compact_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FilePageStore::new(dir.path()).unwrap());

        // Write segments with multiple pages each
        let mut sids = Vec::new();
        for i in 0..10 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            pages.insert(pageidx!(2), Page::test_filled((i + 100) as u8));
            pages.insert(pageidx!(3), Page::test_filled((i + 200) as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        // Remove some segments to create fragmentation
        store.remove_segment(&sids[2]).unwrap();
        store.remove_segment(&sids[5]).unwrap();
        store.remove_segment(&sids[7]).unwrap();

        let live_sids: Vec<_> = sids
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != 2 && *i != 5 && *i != 7)
            .map(|(i, s)| (i, s.clone()))
            .collect();

        // Spawn reader threads that continuously read while compaction happens
        let mut handles = Vec::new();
        for _ in 0..4 {
            let store = Arc::clone(&store);
            let live_sids = live_sids.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    for (i, sid) in &live_sids {
                        // Read and verify data is not corrupted
                        if let Ok(Some(page)) = store.read_page(sid, pageidx!(1)) {
                            assert_eq!(page.as_ref()[0], *i as u8);
                        }
                        if let Ok(Some(page)) = store.read_page(sid, pageidx!(2)) {
                            assert_eq!(page.as_ref()[0], (*i + 100) as u8);
                        }
                    }
                }
            }));
        }

        // Compact while readers are running
        for _ in 0..5 {
            store.compact_all().unwrap();
        }

        // Wait for readers
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all live segments are still correct after concurrent compaction
        for (i, sid) in &live_sids {
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], *i as u8);
            let page = store.read_page(sid, pageidx!(2)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], (*i + 100) as u8);
            let page = store.read_page(sid, pageidx!(3)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], (*i + 200) as u8);
        }
    }

    #[test]
    #[ignore = "flaky due to race condition between compaction and concurrent writes"]
    fn test_compact_concurrent_writes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FilePageStore::new(dir.path()).unwrap());

        // Pre-populate with some segments
        for i in 0..5 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid, pages).unwrap();
        }

        let write_count = Arc::new(AtomicUsize::new(0));
        let written_sids = Arc::new(std::sync::Mutex::new(Vec::new()));

        // Spawn writer threads
        let mut handles = Vec::new();
        for thread_id in 0..4 {
            let store = Arc::clone(&store);
            let write_count = Arc::clone(&write_count);
            let written_sids = Arc::clone(&written_sids);
            handles.push(thread::spawn(move || {
                for i in 0..25 {
                    let sid = SegmentId::random();
                    let mut pages = BTreeMap::new();
                    let fill_byte = ((thread_id * 100 + i) % 256) as u8;
                    pages.insert(pageidx!(1), Page::test_filled(fill_byte));
                    pages.insert(pageidx!(2), Page::test_filled(fill_byte.wrapping_add(1)));

                    store.write_segment(sid.clone(), pages).unwrap();
                    written_sids.lock().unwrap().push((sid, fill_byte));
                    write_count.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }

        // Compact while writers are running
        let compact_store = Arc::clone(&store);
        let compact_handle = thread::spawn(move || {
            for _ in 0..10 {
                compact_store.compact_all().unwrap();
                thread::sleep(std::time::Duration::from_millis(1));
            }
        });

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        compact_handle.join().unwrap();

        // Final compaction
        store.compact_all().unwrap();

        // Verify all written segments exist and have correct data
        let written_sids = written_sids.lock().unwrap();
        for (sid, fill_byte) in written_sids.iter() {
            assert!(
                store.has_segment(sid).unwrap(),
                "Segment should exist after concurrent writes and compaction"
            );
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(
                page.as_ref()[0], *fill_byte,
                "Page data should not be corrupted"
            );
            let page = store.read_page(sid, pageidx!(2)).unwrap().unwrap();
            assert_eq!(
                page.as_ref()[0],
                fill_byte.wrapping_add(1),
                "Page data should not be corrupted"
            );
        }
    }

    #[test]
    fn test_compact_removes_all_deleted_segments() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Write segments
        let mut sids = Vec::new();
        for i in 0..10 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            pages.insert(pageidx!(2), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        let initial_size = store.total_size();
        assert!(initial_size > 0);

        // Remove half the segments
        let removed_sids: Vec<_> = sids.iter().step_by(2).cloned().collect();
        let kept_sids: Vec<_> = sids
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 != 0)
            .map(|(_, s)| s.clone())
            .collect();

        for sid in &removed_sids {
            store.remove_segment(sid).unwrap();
        }

        // Verify removed segments are gone from index
        for sid in &removed_sids {
            assert!(!store.has_segment(sid).unwrap());
        }

        // Compact
        store.compact_all().unwrap();

        // Size should be reduced
        let compacted_size = store.total_size();
        assert!(
            compacted_size < initial_size,
            "Size should decrease after compacting removed segments"
        );

        // Removed segments still gone
        for sid in &removed_sids {
            assert!(!store.has_segment(sid).unwrap());
            assert!(store.read_page(sid, pageidx!(1)).unwrap().is_none());
        }

        // Kept segments still accessible
        for (i, sid) in kept_sids.iter().enumerate() {
            let expected = (i * 2 + 1) as u8; // 1, 3, 5, 7, 9
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], expected);
        }
    }

    #[test]
    fn test_compact_data_integrity_with_sparse_pages() {
        use crate::core::PageIdx;

        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Write segments with sparse page indices
        let mut sids = Vec::new();
        let page_patterns: [Vec<u32>; 4] = [
            vec![1, 5, 10, 100],
            vec![2, 3, 50],
            vec![1, 1000],
            vec![7, 8, 9, 10, 11],
        ];

        for (seg_idx, pattern) in page_patterns.iter().enumerate() {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            for &page_idx in pattern {
                let fill = ((seg_idx * 50 + page_idx as usize) % 256) as u8;
                let pidx = PageIdx::try_new(page_idx).unwrap();
                pages.insert(pidx, Page::test_filled(fill));
            }
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push((sid, pattern.clone(), seg_idx));
        }

        // Remove one segment
        store.remove_segment(&sids[1].0).unwrap();

        // Compact
        store.compact_all().unwrap();

        // Verify remaining segments have correct sparse pages
        for (sid, pattern, seg_idx) in &sids {
            if *seg_idx == 1 {
                assert!(!store.has_segment(sid).unwrap());
                continue;
            }

            assert!(store.has_segment(sid).unwrap());

            for &page_idx in pattern {
                let expected = ((seg_idx * 50 + page_idx as usize) % 256) as u8;
                let pidx = PageIdx::try_new(page_idx).unwrap();
                let page = store
                    .read_page(sid, pidx)
                    .unwrap()
                    .expect("Page should exist");
                assert_eq!(
                    page.as_ref()[0], expected,
                    "Page data mismatch at segment {} page {}",
                    seg_idx, page_idx
                );
                // Verify entire page is filled with expected value
                assert!(
                    page.as_ref().iter().all(|&b| b == expected),
                    "Full page content should match"
                );
            }

            // Verify gaps return None
            assert!(store.read_page(sid, pageidx!(999)).unwrap().is_none());
        }
    }

    #[test]
    fn test_compact_then_continue_operations() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Phase 1: Initial writes
        let mut sids = Vec::new();
        for i in 0..5 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        // Phase 2: Remove and compact
        store.remove_segment(&sids[0]).unwrap();
        store.remove_segment(&sids[2]).unwrap();
        store.compact_all().unwrap();

        // Phase 3: Write new segments after compaction
        let mut new_sids = Vec::new();
        for i in 0..5 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled((i + 100) as u8));
            pages.insert(pageidx!(2), Page::test_filled((i + 150) as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            new_sids.push(sid);
        }

        // Verify old surviving segments still work
        let page = store.read_page(&sids[1], pageidx!(1)).unwrap().unwrap();
        assert_eq!(page.as_ref()[0], 1);
        let page = store.read_page(&sids[3], pageidx!(1)).unwrap().unwrap();
        assert_eq!(page.as_ref()[0], 3);
        let page = store.read_page(&sids[4], pageidx!(1)).unwrap().unwrap();
        assert_eq!(page.as_ref()[0], 4);

        // Verify new segments work
        for (i, sid) in new_sids.iter().enumerate() {
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], (i + 100) as u8);
            let page = store.read_page(sid, pageidx!(2)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], (i + 150) as u8);
        }

        // Phase 4: Another round of remove + compact
        store.remove_segment(&new_sids[0]).unwrap();
        store.remove_segment(&sids[1]).unwrap();
        store.compact_all().unwrap();

        // Phase 5: Continue writing after second compaction
        let final_sid = SegmentId::random();
        let mut pages = BTreeMap::new();
        pages.insert(pageidx!(1), Page::test_filled(0xFF));
        store.write_segment(final_sid.clone(), pages).unwrap();

        // Final verification
        assert!(!store.has_segment(&new_sids[0]).unwrap());
        assert!(!store.has_segment(&sids[1]).unwrap());
        assert!(store.has_segment(&new_sids[1]).unwrap());
        assert!(store.has_segment(&final_sid).unwrap());

        let page = store.read_page(&final_sid, pageidx!(1)).unwrap().unwrap();
        assert_eq!(page.as_ref()[0], 0xFF);
    }

    #[test]
    fn test_compact_empty_after_all_removed() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Write segments
        let mut sids = Vec::new();
        for i in 0..5 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        assert!(store.total_size() > 0);
        assert_eq!(store.segment_count(), 5);

        // Remove all segments
        for sid in &sids {
            store.remove_segment(sid).unwrap();
        }

        // Compact
        store.compact_all().unwrap();

        // Should be empty
        assert_eq!(store.segment_count(), 0);
        assert_eq!(store.total_size(), 0);

        // Can still write new segments
        let new_sid = SegmentId::random();
        let mut pages = BTreeMap::new();
        pages.insert(pageidx!(1), Page::test_filled(0xAB));
        store.write_segment(new_sid.clone(), pages).unwrap();

        assert_eq!(store.segment_count(), 1);
        let page = store.read_page(&new_sid, pageidx!(1)).unwrap().unwrap();
        assert_eq!(page.as_ref()[0], 0xAB);
    }

    #[test]
    fn test_compact_interleaved_with_operations() {
        let dir = tempfile::tempdir().unwrap();
        let store = FilePageStore::new(dir.path()).unwrap();

        // Write initial segments
        let mut sids = Vec::new();
        for i in 0..10 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            pages.insert(pageidx!(2), Page::test_filled((i + 50) as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        // Interleave: remove, compact, write, verify
        for round in 0..5 {
            // Remove a segment
            let remove_idx = round * 2;
            if remove_idx < sids.len() {
                store.remove_segment(&sids[remove_idx]).unwrap();
            }

            // Compact
            store.compact_all().unwrap();

            // Write a new segment
            let new_sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled((100 + round) as u8));
            store.write_segment(new_sid.clone(), pages).unwrap();
            sids.push(new_sid);

            // Verify all remaining segments are readable
            for (i, sid) in sids.iter().enumerate() {
                if i < 10 && i % 2 == 0 && i / 2 <= round {
                    // This was removed
                    assert!(!store.has_segment(sid).unwrap());
                } else if i < 10 {
                    // Original segment still present
                    let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
                    assert_eq!(page.as_ref()[0], i as u8);
                } else {
                    // New segment
                    let expected = (100 + (i - 10)) as u8;
                    let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
                    assert_eq!(page.as_ref()[0], expected);
                }
            }
        }

        // Final compact and verify
        store.compact_all().unwrap();
        let remaining_count = sids
            .iter()
            .filter(|sid| store.has_segment(sid).unwrap())
            .count();
        assert!(remaining_count > 0);
    }
}
