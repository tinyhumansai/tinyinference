//! Private wire records that are not part of the public provider contract.

use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct FileList {
    pub data: Vec<FileRecord>,
}

#[derive(Deserialize)]
pub(super) struct FileRecord {
    pub id: String,
    pub filename: String,
    pub bytes: u64,
    pub created_at: u64,
}
