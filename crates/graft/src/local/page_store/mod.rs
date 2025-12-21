//! Page storage backends for storing segment pages locally.
//!
//! This module provides a trait [`PageStore`] and implementations for storing
//! pages locally. Pages are 16KB blobs keyed by (SegmentId, PageIdx).
//!
//! Two implementations are provided:
//! - [`FilePageStore`]: File-based storage with one file per segment
//! - [`FjallPageStore`]: Wrapper around fjall's pages keyspace

mod file_store;
mod fjall_store;

pub use file_store::FilePageStore;
pub use fjall_store::FjallPageStore;

use std::collections::BTreeMap;

use crate::core::{PageIdx, SegmentId, page::Page};
use crate::local::fjall_storage::FjallStorageErr;

/// Error type for page store operations
#[derive(Debug, thiserror::Error)]
pub enum PageStoreErr {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid segment file: {0}")]
    InvalidSegmentFile(String),

    #[error("Page not found: {sid}/{pageidx}")]
    PageNotFound { sid: SegmentId, pageidx: PageIdx },

    #[error("Fjall error: {0}")]
    Fjall(#[from] fjall::Error),

    #[error("Fjall storage error: {0}")]
    FjallStorage(#[from] FjallStorageErr),
}

/// Trait for page storage backends.
///
/// Pages are write-once 16KB blobs, keyed by (SegmentId, PageIdx).
/// Segments are written atomically as a batch of pages.
pub trait PageStore: Send + Sync {
    /// Check if a specific page exists in the store.
    fn has_page(&self, sid: &SegmentId, pageidx: PageIdx) -> Result<bool, PageStoreErr>;

    /// Read a single page from the store.
    /// Returns None if the page doesn't exist.
    fn read_page(&self, sid: &SegmentId, pageidx: PageIdx) -> Result<Option<Page>, PageStoreErr>;

    /// Write an entire segment atomically.
    /// Pages must be provided in PageIdx order.
    fn write_segment(
        &self,
        sid: SegmentId,
        pages: BTreeMap<PageIdx, Page>,
    ) -> Result<(), PageStoreErr>;

    /// Check if a segment exists (has any pages).
    fn has_segment(&self, sid: &SegmentId) -> Result<bool, PageStoreErr>;

    /// Remove a segment and all its pages from the store.
    /// This is used to clean up old segments after they've been superseded.
    /// Returns Ok(()) even if the segment doesn't exist.
    fn remove_segment(&self, sid: &SegmentId) -> Result<(), PageStoreErr>;
}
