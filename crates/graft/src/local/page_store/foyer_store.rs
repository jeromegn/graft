//! Foyer-based page store with mmap'd active pack and lazy indexing.
//!
//! Architecture:
//! - Segments are batched into pack files by segment ID prefix (first 8 chars)
//! - Active pack is mmap'd for fast writes (memcpy instead of syscall)
//! - Cold packs are scanned lazily on first access
//! - Pages are cached in foyer's in-memory LRU cache
//! - No fsync on writes for performance (durability via remote storage)
//! - Thread-safe via RwLock
//!
//! Pack file format (e.g., `8H24WW94.pack`):
//! ```text
//! ┌─────────────────────────────────────┐
//! │ Segment entry 1                     │
//! │   Magic: "GSEG"          (4 bytes)  │
//! │   Segment ID length      (1 byte)   │
//! │   Segment ID             (N bytes)  │
//! │   Entry size             (4 bytes)  │ ← splinter + pages size
//! │   Splinter size          (4 bytes)  │
//! │   Splinter bitmap        (variable) │
//! │   Page data              (M × 16KB) │
//! ├─────────────────────────────────────┤
//! │ Segment entry 2 ...                 │
//! └─────────────────────────────────────┘
//! ```

use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::mpsc::{self, Sender},
    thread::{self, JoinHandle},
};

use bytes::Bytes;
use foyer::{Cache, CacheBuilder};
use memmap2::MmapMut;
use parking_lot::RwLock;
use splinter_rs::{CowSplinter, PartitionRead};

use crate::core::{
    PageIdx, SegmentId,
    page::{PAGESIZE, Page},
    pageset::PageSet,
};

use super::{PageStore, PageStoreErr};

/// Magic bytes identifying a segment entry in a pack file
const SEGMENT_MAGIC: &[u8; 4] = b"GSEG";

/// Number of characters from segment ID to use as pack file prefix
const PACK_PREFIX_LEN: usize = 8;

/// Default maximum size for active pack before rotation (64 MiB)
const DEFAULT_PACK_CAPACITY: usize = 64 * 1024 * 1024;

/// Configuration for FoyerStore.
#[derive(Debug, Clone)]
pub struct FoyerStoreConfig {
    /// In-memory cache capacity in bytes.
    pub memory_capacity: usize,
    /// Maximum size of each pack file before rotation.
    pub pack_capacity: usize,
}

impl Default for FoyerStoreConfig {
    fn default() -> Self {
        Self {
            memory_capacity: 64 * 1024 * 1024, // 64 MiB
            pack_capacity: DEFAULT_PACK_CAPACITY,
        }
    }
}

impl FoyerStoreConfig {
    /// Create a new config with specified memory capacity.
    pub fn new(memory_capacity: usize) -> Self {
        Self {
            memory_capacity,
            pack_capacity: DEFAULT_PACK_CAPACITY,
        }
    }

    /// Set the pack capacity.
    pub fn with_pack_capacity(mut self, capacity: usize) -> Self {
        self.pack_capacity = capacity;
        self
    }
}

/// Cache key combining SegmentId and PageIdx.
#[derive(Clone, PartialEq, Eq, Hash)]
struct PageKey {
    sid: SegmentId,
    pageidx: PageIdx,
}

impl PageKey {
    fn new(sid: SegmentId, pageidx: PageIdx) -> Self {
        Self { sid, pageidx }
    }
}

/// Location of a segment within a pack file.
struct SegmentLocation {
    /// Which pack file contains this segment
    pack_prefix: String,
    /// Byte offset where page data starts (after header + splinter)
    data_offset: u64,
    /// Page set (which pages exist)
    pageset: PageSet,
}

/// Active pack file with mmap for fast writes.
struct ActivePack {
    /// Pack file prefix
    prefix: String,
    /// Open file handle (kept for extending without reopen)
    file: File,
    /// Memory-mapped region for writes
    mmap: MmapMut,
    /// Current write offset
    offset: usize,
    /// Total capacity (mmap size)
    capacity: usize,
}

/// Read-only pack file handle.
struct ReadPack {
    /// Open file handle for pread
    file: File,
    /// Whether this pack has been scanned/indexed
    scanned: bool,
}

/// Interior mutable state protected by RwLock.
struct StoreInner {
    /// Segment location index
    segments: HashMap<SegmentId, SegmentLocation>,
    /// Read-only pack file handles (prefix -> pack)
    packs: HashMap<String, ReadPack>,
    /// Currently active pack for writes
    active: Option<ActivePack>,
    /// Pack capacity for rotation
    pack_capacity: usize,
}

/// Page store using foyer in-memory cache backed by pack file storage.
///
/// Thread-safe via RwLock - reads can be concurrent, writes are exclusive.
pub struct FoyerStore {
    /// Root directory for pack files
    root: PathBuf,
    /// In-memory page cache (foyer is internally synchronized)
    cache: Cache<PageKey, Bytes>,
    /// Mutable state protected by RwLock
    inner: RwLock<StoreInner>,
    /// Channel to send mmaps for background cleanup
    drop_sender: Sender<MmapMut>,
    /// Background thread handle for cleanup (joined on drop)
    _drop_thread: JoinHandle<()>,
}

impl FoyerStore {
    /// Create a new FoyerStore with default configuration.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, PageStoreErr> {
        Self::with_config(path, FoyerStoreConfig::default())
    }

    /// Create a new FoyerStore with custom configuration.
    pub fn with_config(path: impl AsRef<Path>, config: FoyerStoreConfig) -> Result<Self, PageStoreErr> {
        let root = path.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;

        let cache = CacheBuilder::new(config.memory_capacity)
            .with_weighter(|_key: &PageKey, value: &Bytes| {
                value.len() + 20
            })
            .build();

        // Create channel for background mmap cleanup
        let (drop_sender, drop_receiver) = mpsc::channel::<MmapMut>();

        // Spawn background thread for cleanup (munmap/close are slow)
        let drop_thread = thread::Builder::new()
            .name("foyer-drop".to_string())
            .spawn(move || {
                while let Ok(mmap) = drop_receiver.recv() {
                    drop(mmap);
                }
            })
            .expect("failed to spawn foyer-drop thread");

        let store = Self {
            root,
            cache,
            inner: RwLock::new(StoreInner {
                segments: HashMap::new(),
                packs: HashMap::new(),
                active: None,
                pack_capacity: config.pack_capacity,
            }),
            drop_sender,
            _drop_thread: drop_thread,
        };

        Ok(store)
    }

    /// Get pack file prefix from segment ID
    #[inline]
    fn pack_prefix(sid: &SegmentId) -> String {
        let sid_str = sid.to_string();
        sid_str.chars().take(PACK_PREFIX_LEN).collect()
    }

    /// Get the path for a pack file
    #[inline]
    fn pack_path(&self, prefix: &str) -> PathBuf {
        self.root.join(format!("{prefix}.pack"))
    }

    /// Scan a pack file and index all segments within it.
    /// Caller must hold write lock.
    fn scan_pack_file_locked(&self, inner: &mut StoreInner, prefix: &str) -> Result<(), PageStoreErr> {
        let path = self.pack_path(prefix);
        if !path.exists() {
            return Ok(());
        }

        let mut file = File::open(&path)?;
        let file_len = file.metadata()?.len();
        if file_len == 0 {
            return Ok(());
        }

        // Read entire pack file for scanning
        file.seek(SeekFrom::Start(0))?;
        let mut data = vec![0u8; file_len as usize];
        file.read_exact(&mut data)?;

        let mut offset = 0usize;

        while offset + 13 <= data.len() {
            if &data[offset..offset + 4] != SEGMENT_MAGIC {
                break;
            }

            let sid_len = data[offset + 4] as usize;
            if offset + 13 + sid_len > data.len() {
                break;
            }

            let sid_bytes = &data[offset + 5..offset + 5 + sid_len];
            let sid_str = std::str::from_utf8(sid_bytes)
                .map_err(|e| PageStoreErr::InvalidSegmentFile(format!("bad segment id: {e}")))?;
            let sid: SegmentId = sid_str
                .parse()
                .map_err(|e| PageStoreErr::InvalidSegmentFile(format!("bad segment id: {e}")))?;

            let entry_size_offset = offset + 5 + sid_len;
            let entry_size = u32::from_le_bytes(
                data[entry_size_offset..entry_size_offset + 4].try_into().unwrap(),
            ) as usize;

            let splinter_size_offset = entry_size_offset + 4;
            let splinter_size = u32::from_le_bytes(
                data[splinter_size_offset..splinter_size_offset + 4].try_into().unwrap(),
            ) as usize;

            let splinter_start = splinter_size_offset + 4;
            let splinter_end = splinter_start + splinter_size;

            if splinter_end > data.len() {
                break;
            }

            let splinter_bytes = Bytes::copy_from_slice(&data[splinter_start..splinter_end]);
            let splinter = CowSplinter::from_bytes(splinter_bytes)
                .map_err(|e| PageStoreErr::InvalidSegmentFile(format!("bad splinter: {e}")))?;
            let pageset = PageSet::new(splinter);

            inner.segments.insert(sid, SegmentLocation {
                pack_prefix: prefix.to_string(),
                data_offset: splinter_end as u64,
                pageset,
            });

            let header_size = 4 + 1 + sid_len + 4 + 4;
            offset += header_size + entry_size;
        }

        // Open file for reads and mark as scanned
        let read_file = OpenOptions::new().read(true).open(&path)?;
        inner.packs.insert(prefix.to_string(), ReadPack {
            file: read_file,
            scanned: true,
        });

        Ok(())
    }

    /// Create a new active pack for the given prefix.
    /// Caller must hold write lock.
    fn create_active_pack_locked(&self, inner: &mut StoreInner, prefix: &str, min_additional: usize) -> Result<(), PageStoreErr> {
        let path = self.pack_path(prefix);

        // Create and pre-allocate the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let current_len = file.metadata()?.len() as usize;

        // Capacity must be at least current_len + min_additional, rounded up to pack_capacity
        let min_capacity = current_len + min_additional;
        let target_capacity = if min_capacity <= inner.pack_capacity {
            inner.pack_capacity
        } else {
            // Round up to next multiple of pack_capacity
            ((min_capacity + inner.pack_capacity - 1) / inner.pack_capacity) * inner.pack_capacity
        };

        // Pre-allocate to required capacity (may already be larger)
        let final_capacity = current_len.max(target_capacity);
        if file.metadata()?.len() < final_capacity as u64 {
            file.set_len(final_capacity as u64)?;
        }

        // mmap the file
        // SAFETY: Protected by RwLock write lock
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // Use actual mmap size as capacity
        let actual_capacity = mmap.len();

        inner.active = Some(ActivePack {
            prefix: prefix.to_string(),
            file,
            mmap,
            offset: current_len,
            capacity: actual_capacity,
        });

        Ok(())
    }

    /// Extend the active pack to fit additional data.
    /// Caller must hold write lock.
    fn extend_active_pack_locked(&self, inner: &mut StoreInner, min_additional: usize) -> Result<(), PageStoreErr> {
        let active = inner.active.take().ok_or_else(|| {
            PageStoreErr::InvalidSegmentFile("no active pack to extend".to_string())
        })?;

        let prefix = active.prefix;
        let file = active.file;
        let offset = active.offset;

        // Send mmap to background thread for cleanup (munmap is slow)
        let _ = self.drop_sender.send(active.mmap);

        // Calculate new capacity - round up to next multiple of pack_capacity
        let min_capacity = offset + min_additional;
        let new_capacity = ((min_capacity + inner.pack_capacity - 1) / inner.pack_capacity) * inner.pack_capacity;

        // Extend file (reusing existing handle)
        file.set_len(new_capacity as u64)?;

        // Re-mmap with new size
        // SAFETY: Protected by RwLock write lock
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        // Use actual mmap size as capacity
        let actual_capacity = mmap.len();

        inner.active = Some(ActivePack {
            prefix,
            file,
            mmap,
            offset,
            capacity: actual_capacity,
        });

        Ok(())
    }

    /// Finalize the current active pack (truncate to actual size, move to read packs).
    /// Caller must hold write lock.
    fn finalize_active_pack_locked(&self, inner: &mut StoreInner) -> Result<(), PageStoreErr> {
        if let Some(active) = inner.active.take() {
            // Send mmap to background thread for cleanup (munmap is slow)
            let _ = self.drop_sender.send(active.mmap);

            // Truncate file to actual used size (reuse existing handle)
            active.file.set_len(active.offset as u64)?;

            // Reuse file handle for reads (already open for read+write)
            inner.packs.insert(active.prefix, ReadPack {
                file: active.file,
                scanned: true,
            });
        }

        Ok(())
    }

    /// Get the number of segments in the store.
    pub fn segment_count(&self) -> usize {
        self.inner.read().segments.len()
    }

    /// Get cache statistics: (used_weight, capacity)
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.cache.usage(), self.cache.capacity())
    }

    /// Clear the in-memory cache (data remains on disk)
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Access a page's data via callback.
    pub fn with_page<F, R>(
        &self,
        sid: &SegmentId,
        pageidx: PageIdx,
        f: F,
    ) -> Result<Option<R>, PageStoreErr>
    where
        F: FnOnce(&[u8]) -> R,
    {
        match self.read_page(sid, pageidx)? {
            Some(page) => Ok(Some(f(page.as_ref()))),
            None => Ok(None),
        }
    }
}

impl PageStore for FoyerStore {
    fn has_page(&self, sid: &SegmentId, pageidx: PageIdx) -> Result<bool, PageStoreErr> {
        let prefix = Self::pack_prefix(sid);

        // Try read lock first for fast path
        {
            let inner = self.inner.read();
            if let Some(loc) = inner.segments.get(sid) {
                return Ok(loc.pageset.contains(pageidx));
            }

            // Check if in active pack but not indexed (shouldn't happen)
            if let Some(ref active) = inner.active {
                if prefix == active.prefix {
                    return Ok(false);
                }
            }

            // Already scanned this pack?
            if let Some(pack) = inner.packs.get(&prefix) {
                if pack.scanned {
                    return Ok(false);
                }
            }
        }

        // Need to scan pack - upgrade to write lock
        let mut inner = self.inner.write();

        // Double-check after acquiring write lock
        if inner.segments.contains_key(sid) {
            return Ok(inner.segments.get(sid).unwrap().pageset.contains(pageidx));
        }

        // Scan the pack
        self.scan_pack_file_locked(&mut inner, &prefix)?;

        Ok(inner.segments.get(sid).is_some_and(|loc| loc.pageset.contains(pageidx)))
    }

    fn read_page(&self, sid: &SegmentId, pageidx: PageIdx) -> Result<Option<Page>, PageStoreErr> {
        let key = PageKey::new(sid.clone(), pageidx);

        // Try cache first (no lock needed - foyer is thread-safe)
        if let Some(entry) = self.cache.get(&key) {
            let bytes = entry.value().clone();
            return Ok(Some(
                Page::try_from(bytes).map_err(|e| PageStoreErr::InvalidSegmentFile(e.to_string()))?,
            ));
        }

        let prefix = Self::pack_prefix(sid);

        // Try to read with read lock first
        {
            let inner = self.inner.read();

            // Check if segment is indexed
            if let Some(location) = inner.segments.get(sid) {
                if !location.pageset.contains(pageidx) {
                    return Ok(None);
                }

                // Calculate page offset
                let rank = location.pageset.splinter().rank(pageidx.to_u32()) - 1;
                let page_offset = location.data_offset + (rank * PAGESIZE.as_usize()) as u64;

                // Check if it's in the active pack
                if let Some(ref active) = inner.active {
                    if location.pack_prefix == active.prefix {
                        let start = page_offset as usize;
                        let end = start.saturating_add(PAGESIZE.as_usize());
                        if start < active.mmap.len() && end <= active.mmap.len() {
                            let bytes = Bytes::copy_from_slice(&active.mmap[start..end]);
                            self.cache.insert(key, bytes.clone());
                            return Ok(Some(
                                Page::try_from(bytes).map_err(|e| PageStoreErr::InvalidSegmentFile(e.to_string()))?,
                            ));
                        }
                    }
                }

                // Try to read from packs
                if let Some(pack) = inner.packs.get(&location.pack_prefix) {
                    let mut buf = vec![0u8; PAGESIZE.as_usize()];
                    pack.file.read_at(&mut buf, page_offset)?;
                    let bytes = Bytes::copy_from_slice(&buf);
                    self.cache.insert(key, bytes.clone());
                    return Ok(Some(
                        Page::try_from(bytes).map_err(|e| PageStoreErr::InvalidSegmentFile(e.to_string()))?,
                    ));
                }
            }

            // Check if pack needs scanning
            if inner.active.as_ref().is_some_and(|a| a.prefix == prefix) {
                return Ok(None);
            }

            if inner.packs.get(&prefix).is_some_and(|p| p.scanned) {
                return Ok(None);
            }
        }

        // Need to scan pack or open pack file - upgrade to write lock
        let mut inner = self.inner.write();

        // Double-check segment
        if !inner.segments.contains_key(sid) {
            // Scan the pack
            self.scan_pack_file_locked(&mut inner, &prefix)?;
        }

        // Now try to read
        let location = match inner.segments.get(sid) {
            Some(loc) => loc,
            None => return Ok(None),
        };

        if !location.pageset.contains(pageidx) {
            return Ok(None);
        }

        let rank = location.pageset.splinter().rank(pageidx.to_u32()) - 1;
        let page_offset = location.data_offset + (rank * PAGESIZE.as_usize()) as u64;
        let pack_prefix = location.pack_prefix.clone();

        // Open pack file if needed
        if !inner.packs.contains_key(&pack_prefix) {
            let path = self.pack_path(&pack_prefix);
            if path.exists() {
                let read_file = OpenOptions::new().read(true).open(&path)?;
                inner.packs.insert(pack_prefix.clone(), ReadPack {
                    file: read_file,
                    scanned: true,
                });
            }
        }

        // Read from pack using local buffer to avoid borrow conflict
        let pack = inner.packs.get(&pack_prefix).ok_or_else(|| {
            PageStoreErr::InvalidSegmentFile(format!("pack not found: {}", pack_prefix))
        })?;

        let mut buf = vec![0u8; PAGESIZE.as_usize()];
        pack.file.read_at(&mut buf, page_offset)?;
        let bytes = Bytes::from(buf);

        self.cache.insert(key, bytes.clone());

        Ok(Some(
            Page::try_from(bytes).map_err(|e| PageStoreErr::InvalidSegmentFile(e.to_string()))?,
        ))
    }

    fn write_segment(
        &self,
        sid: SegmentId,
        pages: BTreeMap<PageIdx, Page>,
    ) -> Result<(), PageStoreErr> {
        if pages.is_empty() {
            return Ok(());
        }

        let prefix = Self::pack_prefix(&sid);

        // Build pageset and calculate sizes
        let pageset = PageSet::from(splinter_rs::Splinter::from_iter(
            pages.keys().map(|idx| idx.to_u32()),
        ));
        let splinter_bytes = pageset.splinter().encode_to_bytes();

        let sid_str = sid.to_string();
        let sid_bytes = sid_str.as_bytes();

        let header_size = 4 + 1 + sid_bytes.len() + 4 + 4;
        let entry_size = splinter_bytes.len() + pages.len() * PAGESIZE.as_usize();
        let total_size = header_size + entry_size;

        // Acquire write lock for the entire write operation
        let mut inner = self.inner.write();

        // Check if we need to create or rotate the active pack
        let (need_new_pack, same_prefix, has_active) = match &inner.active {
            None => (true, false, false),
            Some(active) => {
                let needs_new = active.prefix != prefix || active.offset + total_size > active.capacity;
                let same = active.prefix == prefix;
                (needs_new, same, true)
            }
        };

        if need_new_pack {
            if same_prefix {
                self.extend_active_pack_locked(&mut inner, total_size)?;
            } else {
                if has_active {
                    self.finalize_active_pack_locked(&mut inner)?;
                }
                self.create_active_pack_locked(&mut inner, &prefix, total_size)?;
            }
        }

        let active = inner.active.as_mut().ok_or_else(|| {
            PageStoreErr::InvalidSegmentFile("no active pack after creation".to_string())
        })?;
        let offset = active.offset;

        // Bounds check before writing
        let end_offset = offset + total_size;
        if end_offset > active.mmap.len() {
            return Err(PageStoreErr::InvalidSegmentFile(format!(
                "write would exceed mmap bounds: offset={}, total_size={}, mmap_len={}, capacity={}",
                offset, total_size, active.mmap.len(), active.capacity
            )));
        }

        // Write header: magic
        active.mmap[offset..offset + 4].copy_from_slice(SEGMENT_MAGIC);

        // Write header: sid_len + sid
        active.mmap[offset + 4] = sid_bytes.len() as u8;
        active.mmap[offset + 5..offset + 5 + sid_bytes.len()].copy_from_slice(sid_bytes);

        // Write header: entry_size + splinter_size
        let entry_size_offset = offset + 5 + sid_bytes.len();
        active.mmap[entry_size_offset..entry_size_offset + 4]
            .copy_from_slice(&(entry_size as u32).to_le_bytes());
        active.mmap[entry_size_offset + 4..entry_size_offset + 8]
            .copy_from_slice(&(splinter_bytes.len() as u32).to_le_bytes());

        // Write splinter
        let splinter_offset = entry_size_offset + 8;
        active.mmap[splinter_offset..splinter_offset + splinter_bytes.len()]
            .copy_from_slice(&splinter_bytes);

        // Write pages
        let mut page_offset = splinter_offset + splinter_bytes.len();
        for page in pages.values() {
            active.mmap[page_offset..page_offset + PAGESIZE.as_usize()]
                .copy_from_slice(page.as_ref());
            page_offset += PAGESIZE.as_usize();
        }

        let data_offset = splinter_offset + splinter_bytes.len();
        active.offset = page_offset;

        // Update segment index
        inner.segments.insert(sid.clone(), SegmentLocation {
            pack_prefix: prefix,
            data_offset: data_offset as u64,
            pageset,
        });

        // Pre-populate cache (after releasing write lock conceptually, but cache is thread-safe)
        for (pageidx, page) in pages {
            let key = PageKey::new(sid.clone(), pageidx);
            self.cache.insert(key, page.into_bytes());
        }

        Ok(())
    }

    fn has_segment(&self, sid: &SegmentId) -> Result<bool, PageStoreErr> {
        let prefix = Self::pack_prefix(sid);

        // Try read lock first
        {
            let inner = self.inner.read();
            if inner.segments.contains_key(sid) {
                return Ok(true);
            }

            if inner.active.as_ref().is_some_and(|a| a.prefix == prefix) {
                return Ok(false);
            }

            if inner.packs.get(&prefix).is_some_and(|p| p.scanned) {
                return Ok(false);
            }
        }

        // Need to scan - upgrade to write lock
        let mut inner = self.inner.write();

        if inner.segments.contains_key(sid) {
            return Ok(true);
        }

        self.scan_pack_file_locked(&mut inner, &prefix)?;

        Ok(inner.segments.contains_key(sid))
    }

    fn remove_segment(&self, sid: &SegmentId) -> Result<(), PageStoreErr> {
        let mut inner = self.inner.write();
        if let Some(location) = inner.segments.remove(sid) {
            for pageidx in location.pageset.iter() {
                let key = PageKey::new(sid.clone(), pageidx);
                self.cache.remove(&key);
            }
        }
        Ok(())
    }
}

impl Drop for FoyerStore {
    fn drop(&mut self) {
        // Finalize active pack on drop
        let mut inner = self.inner.write();
        let _ = self.finalize_active_pack_locked(&mut inner);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pageidx;

    fn create_test_store() -> (tempfile::TempDir, FoyerStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = FoyerStore::new(dir.path()).unwrap();
        (dir, store)
    }

    #[test]
    fn test_write_and_read_segment() {
        let (_dir, store) = create_test_store();

        let sid = SegmentId::random();
        let mut pages = BTreeMap::new();
        pages.insert(pageidx!(1), Page::test_filled(0x11));
        pages.insert(pageidx!(3), Page::test_filled(0x33));
        pages.insert(pageidx!(5), Page::test_filled(0x55));

        store.write_segment(sid.clone(), pages).unwrap();

        assert!(store.has_segment(&sid).unwrap());
        assert!(store.has_page(&sid, pageidx!(1)).unwrap());
        assert!(!store.has_page(&sid, pageidx!(2)).unwrap());
        assert!(store.has_page(&sid, pageidx!(3)).unwrap());

        let page1 = store.read_page(&sid, pageidx!(1)).unwrap().unwrap();
        assert_eq!(page1.as_ref()[0], 0x11);

        let page3 = store.read_page(&sid, pageidx!(3)).unwrap().unwrap();
        assert_eq!(page3.as_ref()[0], 0x33);

        assert!(store.read_page(&sid, pageidx!(2)).unwrap().is_none());
    }

    #[test]
    fn test_remove_segment() {
        let (_dir, store) = create_test_store();

        let sid = SegmentId::random();
        let mut pages = BTreeMap::new();
        pages.insert(pageidx!(1), Page::test_filled(0x11));
        pages.insert(pageidx!(2), Page::test_filled(0x22));

        store.write_segment(sid.clone(), pages).unwrap();
        assert!(store.has_segment(&sid).unwrap());

        store.remove_segment(&sid).unwrap();
        assert!(!store.has_segment(&sid).unwrap());
        assert!(store.read_page(&sid, pageidx!(1)).unwrap().is_none());
    }

    #[test]
    fn test_nonexistent_segment() {
        let (_dir, store) = create_test_store();

        let sid = SegmentId::random();
        assert!(!store.has_segment(&sid).unwrap());
        assert!(!store.has_page(&sid, pageidx!(1)).unwrap());
        assert!(store.read_page(&sid, pageidx!(1)).unwrap().is_none());
    }

    #[test]
    fn test_multiple_segments_same_pack() {
        let (_dir, store) = create_test_store();

        let mut sids = Vec::new();
        for i in 0..5 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        assert_eq!(store.segment_count(), 5);

        for (i, sid) in sids.iter().enumerate() {
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], i as u8);
        }
    }

    #[test]
    fn test_cache_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let config = FoyerStoreConfig::new(2 * PAGESIZE.as_usize() + 100);
        let store = FoyerStore::with_config(dir.path(), config).unwrap();

        let mut sids = Vec::new();
        for i in 0..5 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        for (i, sid) in sids.iter().enumerate() {
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], i as u8);
        }

        for (i, sid) in sids.iter().enumerate() {
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], i as u8);
        }
    }

    #[test]
    fn test_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let sid = SegmentId::random();

        {
            let store = FoyerStore::new(dir.path()).unwrap();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(0xAB));
            pages.insert(pageidx!(2), Page::test_filled(0xCD));
            store.write_segment(sid.clone(), pages).unwrap();
        }

        {
            let store = FoyerStore::new(dir.path()).unwrap();
            assert!(store.has_segment(&sid).unwrap());

            let page1 = store.read_page(&sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page1.as_ref()[0], 0xAB);

            let page2 = store.read_page(&sid, pageidx!(2)).unwrap().unwrap();
            assert_eq!(page2.as_ref()[0], 0xCD);
        }
    }

    #[test]
    fn test_reopen_pack_after_switch() {
        let dir = tempfile::tempdir().unwrap();
        let store = FoyerStore::new(dir.path()).unwrap();

        let mut sids = Vec::new();
        for i in 0..10 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        store.clear_cache();

        for (i, sid) in sids.iter().enumerate() {
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], i as u8, "segment {i} mismatch");
        }
    }

    #[test]
    fn test_switching_between_packs() {
        let dir = tempfile::tempdir().unwrap();
        let store = FoyerStore::new(dir.path()).unwrap();

        let mut sids = Vec::new();
        for i in 0..5 {
            let sid = SegmentId::random();
            let mut pages = BTreeMap::new();
            pages.insert(pageidx!(1), Page::test_filled(i as u8));
            store.write_segment(sid.clone(), pages).unwrap();
            sids.push(sid);
        }

        for (i, sid) in sids.iter().enumerate() {
            let page = store.read_page(sid, pageidx!(1)).unwrap().unwrap();
            assert_eq!(page.as_ref()[0], i as u8);
        }

        let pack_count = std::fs::read_dir(dir.path())
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .ok()
                    .and_then(|e| e.path().extension().map(|ext| ext == "pack"))
                    .unwrap_or(false)
            })
            .count();

        assert!(pack_count >= 1);
    }
}
