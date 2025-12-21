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
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use arc_swap::ArcSwapOption;
use bytes::Bytes;
use dashmap::DashMap;
use memmap2::Mmap;
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

use crate::core::{
    PageIdx, SegmentId,
    page::{PAGESIZE, Page},
    pageset::PageSet,
};

use super::{PageStore, PageStoreErr};

/// Magic bytes identifying a segment
const MAGIC: &[u8; 8] = b"GRAFTSG\x01";

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
    mmap: Mmap,
    /// The capacity of the mmap (may be larger than actual data)
    capacity: u64,
    /// Segment index: SegmentId -> (offset_in_mmap, data_offset, pageset)
    /// DashMap allows lock-free reads and concurrent inserts without cloning.
    segments: DashMap<SegmentId, (usize, usize, Arc<PageSet>)>,
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
    /// Global index: SegmentId -> shard number (for has_segment)
    /// Uses DashMap for lock-free concurrent access.
    index: DashMap<SegmentId, u8>,
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
            index: DashMap::new(),
        };

        // Rebuild index from existing pack files
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

    /// Rebuild the in-memory index by scanning all pack files
    fn rebuild_index(&self) -> Result<(), PageStoreErr> {
        for shard_num in 0..NUM_SHARDS {
            let config = &self.configs[shard_num];
            if !config.path.exists() {
                continue;
            }

            let metadata = fs::metadata(&config.path)?;
            if metadata.len() == 0 {
                continue;
            }

            // Scan the pack file to find all segments
            let file = File::open(&config.path)?;
            // SAFETY: The file is open and valid. We only read from the mmap.
            // Initialization happens before any concurrent access.
            let mmap = unsafe { Mmap::map(&file)? };
            let capacity = mmap.len() as u64;

            let segments = Self::scan_pack(&mmap, &config.path)?;
            for (sid, _, _, _) in &segments {
                self.index.insert(sid.clone(), shard_num as u8);
            }

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
    /// Format per segment: magic(8) + splinter_size(4) + segment_id(16) + splinter(var) + pages
    fn scan_pack(
        mmap: &[u8],
        _path: &Path,
    ) -> Result<Vec<(SegmentId, usize, usize, PageSet)>, PageStoreErr> {
        let mut segments = Vec::new();
        let mut offset = 0;

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

        // Nothing to load if file doesn't exist or is empty
        if !config.path.exists() {
            return Ok(None);
        }

        // Load the pack file
        let file = File::open(&config.path)?;
        let metadata = file.metadata()?;
        if metadata.len() == 0 {
            return Ok(None);
        }

        // SAFETY: The file is open and valid. We only read from the mmap.
        // Handle EINVAL gracefully - can happen if file was truncated to 0 between
        // our length check and mmap call (race with compaction).
        let mmap = match unsafe { Mmap::map(&file) } {
            Ok(m) => m,
            Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
                // EINVAL from mmap typically means length is 0.
                // Verify the file is actually empty before returning None.
                if file.metadata().map(|m| m.len()).unwrap_or(1) == 0 {
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

impl PageStore for FilePageStore {
    fn has_page(&self, sid: &SegmentId, pageidx: PageIdx) -> Result<bool, PageStoreErr> {
        match self.get_segment(sid)? {
            Some((_, _, _, pageset)) => Ok(pageset.contains(pageidx)),
            None => Ok(false),
        }
    }

    fn read_page(&self, sid: &SegmentId, pageidx: PageIdx) -> Result<Option<Page>, PageStoreErr> {
        let (cached, seg_offset, data_offset, pageset) = match self.get_segment(sid)? {
            Some(s) => s,
            None => return Ok(None),
        };

        if !pageset.contains(pageidx) {
            return Ok(None);
        }

        // Calculate page offset within segment
        let rank = pageset.splinter().rank(pageidx.to_u32()) - 1;
        let page_offset = seg_offset + data_offset + rank * PAGESIZE.as_usize();
        let page_end = page_offset + PAGESIZE.as_usize();

        if page_end > cached.mmap.len() {
            return Err(PageStoreErr::InvalidSegmentFile(
                "page offset beyond file end".to_string(),
            ));
        }

        let page_bytes = Bytes::copy_from_slice(&cached.mmap[page_offset..page_end]);
        let page = Page::try_from(page_bytes).expect("buffer is exactly PAGESIZE");
        Ok(Some(page))
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

        // Extended header: magic + splinter_size + segment_id + splinter
        let sid_bytes = sid.as_bytes();
        let mut header =
            Vec::with_capacity(HEADER_PREFIX_SIZE + SegmentId::SIZE.as_usize() + splinter_bytes.len());
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&splinter_size.to_le_bytes());
        header.extend_from_slice(sid_bytes);
        header.extend_from_slice(&splinter_bytes);

        let data_offset = header.len();
        let segment_size = (data_offset + pages.len() * PAGESIZE.as_usize()) as u64;

        // Ensure writer is initialized
        self.get_or_create_writer(shard_num)?;

        // Get writer (we know it's initialized now)
        let writer_guard = self.writers[shard_num].read();
        let writer = writer_guard.as_ref().unwrap();

        // Atomically reserve space
        let seg_offset = writer.data_size.fetch_add(segment_size, Ordering::SeqCst);
        let new_data_size = seg_offset + segment_size;

        // Check if we need to extend the file and remap
        let current_capacity = self.caches[shard_num]
            .load_full()
            .map_or(0, |c| c.capacity);

        let config = &self.configs[shard_num];

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
                    let result = unsafe { libc::ftruncate(writer.fd, new_capacity as libc::off_t) };
                    if result != 0 {
                        return Err(PageStoreErr::Io(std::io::Error::last_os_error()));
                    }

                    // Create new mmap with larger capacity
                    let file_for_mmap = File::open(&config.path)?;
                    // SAFETY: The file is open and valid. We only read from the mmap.
                    let new_mmap = unsafe { Mmap::map(&file_for_mmap)? };
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
        data_slices.push(&header);
        data_slices.extend(page_refs);

        // Write using pwritev (thread-safe, no seek needed)
        pwritev_all(writer.fd, seg_offset, &data_slices)?;

        // Update global index (lock-free via DashMap)
        self.index.insert(sid.clone(), shard_num as u8);

        // Update cache - insert into DashMap
        if let Some(cached) = self.caches[shard_num].load_full() {
            cached.segments.insert(sid.clone(), (seg_offset as usize, data_offset, Arc::new(pageset)));
        } else {
            // No cache exists yet, need to create one
            let file_for_mmap = File::open(&config.path)?;
            // SAFETY: The file is open and valid. We only read from the mmap.
            let new_mmap = unsafe { Mmap::map(&file_for_mmap)? };
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

        Ok(())
    }

    fn has_segment(&self, sid: &SegmentId) -> Result<bool, PageStoreErr> {
        Ok(self.index.contains_key(sid))
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

    /// Remove a segment from the in-memory index.
    /// The segment data remains in the pack file until compaction.
    pub fn remove_segment(&self, sid: &SegmentId) -> Result<(), PageStoreErr> {
        // Remove from global index (lock-free via DashMap)
        let shard_num = match self.index.remove(sid) {
            Some((_, n)) => n as usize,
            None => return Ok(()), // Already removed
        };

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

    /// Compact a shard by rewriting only live segments.
    /// This reclaims disk space from removed segments.
    /// Uses file locking for multi-process safety.
    pub fn compact_shard(&self, shard_num: u8) -> Result<(), PageStoreErr> {
        let shard_num = shard_num as usize;
        if shard_num >= NUM_SHARDS {
            return Ok(());
        }

        let config = &self.configs[shard_num];

        // Acquire exclusive file lock for compaction
        let _lock = FileLock::exclusive(&config.lock_path)?;

        // Get live segments from cache
        let cached = match self.caches[shard_num].load_full() {
            Some(c) => c,
            None => return Ok(()), // No cache, nothing to compact
        };

        let live_segments: Vec<_> = cached
            .segments
            .iter()
            .map(|entry| {
                let (offset, data_offset, pageset) = entry.value();
                (entry.key().clone(), *offset, *data_offset, pageset.clone())
            })
            .collect();

        if live_segments.is_empty() {
            // All segments removed, truncate the file
            if config.path.exists() {
                OpenOptions::new()
                    .write(true)
                    .open(&config.path)?
                    .set_len(0)?;
            }

            // Reset writer data_size
            if let Some(writer) = self.writers[shard_num].read().as_ref() {
                writer.data_size.store(0, Ordering::SeqCst);
            }

            self.caches[shard_num].store(None);
            self.access_times[shard_num].times.clear();
            self.access_times[shard_num].dirty.store(true, Ordering::Relaxed);
            return Ok(());
        }

        // Read all live segment data from current mmap
        let old_mmap = &cached.mmap;
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

        // Write to new temp file
        let temp_path = config.path.with_extension("pack.tmp");
        let mut temp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;

        let new_segments = DashMap::new();
        let mut new_offset = 0usize;
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

        // Reopen and remap
        let new_file = File::open(&config.path)?;

        // SAFETY: The file is open and valid. We only read from the mmap.
        // We hold the exclusive file lock.
        let new_mmap = unsafe { Mmap::map(&new_file)? };
        let capacity = new_mmap.len() as u64;

        // Update cache atomically
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
        self.index.len()
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
}
