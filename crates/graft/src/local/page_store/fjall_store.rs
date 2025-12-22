//! Fjall-based page storage using an LSM-tree keyspace.
//!
//! This implementation wraps fjall's pages keyspace to provide the same
//! interface as [`FilePageStore`], enabling benchmarking and comparison.

use std::collections::BTreeMap;

use fjall::KvSeparationOptions;

use crate::core::{
    PageIdx, SegmentId,
    page::Page,
};
use crate::local::fjall_storage::{
    fjall_typed::{ReadableExt, TypedKeyspace, WriteBatchExt},
    keys::PageKey,
};

use super::{PageStore, PageStoreErr};

/// Fjall-based page storage.
///
/// Uses an LSM-tree keyspace with KV separation enabled for the large
/// 16KB page values.
pub struct FjallPageStore {
    db: fjall::Database,
    pages: TypedKeyspace<PageKey, Page>,
}

impl FjallPageStore {
    /// Create a new fjall-based page store within the given database.
    pub fn open(db: fjall::Database) -> Result<Self, PageStoreErr> {
        let pages = TypedKeyspace::open(&db, "pages", || {
            fjall::KeyspaceCreateOptions::default()
                .with_kv_separation(Some(KvSeparationOptions::default()))
        })?;

        Ok(Self { db, pages })
    }

    /// Get a reference to the underlying database.
    pub fn db(&self) -> &fjall::Database {
        &self.db
    }

    /// Get a reference to the pages keyspace.
    pub fn keyspace(&self) -> &TypedKeyspace<PageKey, Page> {
        &self.pages
    }
}

impl FjallPageStore {
    /// Access a page's data via callback.
    ///
    /// Note: FjallPageStore doesn't support true zero-copy reads since the
    /// LSM-tree returns owned data. This method reads the page and calls
    /// the callback with a reference to it.
    pub fn with_page<F, R>(
        &self,
        sid: &SegmentId,
        pageidx: PageIdx,
        f: F,
    ) -> Result<Option<R>, PageStoreErr>
    where
        F: FnOnce(&[u8]) -> R,
    {
        let snapshot = self.db.snapshot();
        match snapshot.get_owned(&self.pages, PageKey::new(sid.clone(), pageidx))? {
            Some(page) => Ok(Some(f(page.as_ref()))),
            None => Ok(None),
        }
    }
}

impl PageStore for FjallPageStore {
    fn has_page(&self, sid: &SegmentId, pageidx: PageIdx) -> Result<bool, PageStoreErr> {
        let snapshot = self.db.snapshot();
        Ok(snapshot.contains_key(&self.pages, &PageKey::new(sid.clone(), pageidx))?)
    }

    fn read_page(&self, sid: &SegmentId, pageidx: PageIdx) -> Result<Option<Page>, PageStoreErr> {
        let snapshot = self.db.snapshot();
        Ok(snapshot.get_owned(&self.pages, PageKey::new(sid.clone(), pageidx))?)
    }

    fn write_segment(
        &self,
        sid: SegmentId,
        pages: BTreeMap<PageIdx, Page>,
    ) -> Result<(), PageStoreErr> {
        if pages.is_empty() {
            return Ok(());
        }

        let mut batch = self.db.batch();
        for (pageidx, page) in pages {
            batch.insert_typed(&self.pages, PageKey::new(sid.clone(), pageidx), page);
        }
        batch.commit()?;

        Ok(())
    }

    fn has_segment(&self, sid: &SegmentId) -> Result<bool, PageStoreErr> {
        // Check if any page exists for this segment by doing a prefix scan
        let snapshot = self.db.snapshot();
        let mut iter = snapshot.prefix(&self.pages, sid);
        Ok(iter.next().is_some())
    }

    fn remove_segment(&self, sid: &SegmentId) -> Result<(), PageStoreErr> {
        // Delete all pages with this segment ID prefix
        let snapshot = self.db.snapshot();
        let mut batch = self.db.batch();
        for entry in snapshot.prefix(&self.pages, sid) {
            let (key, _) = entry?;
            batch.remove_typed(&self.pages, key);
        }
        batch.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pageidx;

    #[test]
    fn test_write_and_read_segment() {
        let dir = tempfile::tempdir().unwrap();
        let db = fjall::Database::builder(dir.path()).open().unwrap();
        let store = FjallPageStore::open(db).unwrap();

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
        let db = fjall::Database::builder(dir.path()).open().unwrap();
        let store = FjallPageStore::open(db).unwrap();

        let sid = SegmentId::random();
        assert!(!store.has_segment(&sid).unwrap());
        assert!(!store.has_page(&sid, pageidx!(1)).unwrap());
        assert!(store.read_page(&sid, pageidx!(1)).unwrap().is_none());
    }

    #[test]
    fn test_remove_segment() {
        let dir = tempfile::tempdir().unwrap();
        let db = fjall::Database::builder(dir.path()).open().unwrap();
        let store = FjallPageStore::open(db).unwrap();

        let sid = SegmentId::random();
        let mut pages = BTreeMap::new();
        pages.insert(pageidx!(1), Page::test_filled(0x11));
        pages.insert(pageidx!(3), Page::test_filled(0x33));

        // Write segment
        store.write_segment(sid.clone(), pages).unwrap();
        assert!(store.has_segment(&sid).unwrap());
        assert!(store.has_page(&sid, pageidx!(1)).unwrap());

        // Remove segment
        store.remove_segment(&sid).unwrap();
        assert!(!store.has_segment(&sid).unwrap());
        assert!(!store.has_page(&sid, pageidx!(1)).unwrap());
        assert!(store.read_page(&sid, pageidx!(1)).unwrap().is_none());

        // Removing non-existent segment should not error
        store.remove_segment(&sid).unwrap();
    }
}
