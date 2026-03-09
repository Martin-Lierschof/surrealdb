//! B-tree index iterators for Idx and Uniq indexes.
//!
//! These iterators provide efficient record retrieval using B-tree index structures.
//! They support equality lookups, range scans, compound prefix scans, and union
//! operations.

use anyhow::Result;

use crate::catalog::{DatabaseId, IndexDefinition, NamespaceId};
use crate::exec::index::access_path::RangeBound;
use crate::expr::BinaryOperator;
use crate::idx::planner::ScanDirection;
use crate::key::index::Index;
use crate::kvs::{KVKey, Key, Transaction, Val};
use crate::val::{Array, RecordId, Value};

/// Batch size for index scans.
const INDEX_BATCH_SIZE: u32 = 1000;

fn decode_record_ids(res: Vec<(Key, Val)>) -> Result<Vec<RecordId>> {
	let mut records = Vec::with_capacity(res.len());
	for (_, val) in res {
		let rid: RecordId = revision::from_slice(&val)?;
		records.push(rid);
	}
	Ok(records)
}

/// Iterator for equality lookups on non-unique indexes.
///
/// Scans all records matching a specific key value.
pub(crate) struct IndexEqualIterator {
	/// Current scan position (begin key)
	beg: Vec<u8>,
	/// End key (exclusive)
	end: Vec<u8>,
	/// Whether iteration is complete
	done: bool,
}

impl IndexEqualIterator {
	/// Create a new equality iterator.
	pub(crate) fn new(
		ns: NamespaceId,
		db: DatabaseId,
		ix: &IndexDefinition,
		value: &Value,
	) -> Result<Self> {
		let array = Array::from(vec![value.clone()]);
		let beg = Index::prefix_ids_beg(ns, db, &ix.table_name, ix.index_id, &array)?;
		let end = Index::prefix_ids_end(ns, db, &ix.table_name, ix.index_id, &array)?;
		Ok(Self {
			beg,
			end,
			done: false,
		})
	}

	/// Fetch the next batch of record IDs.
	pub async fn next_batch(&mut self, tx: &Transaction) -> Result<Vec<RecordId>> {
		if self.done {
			return Ok(Vec::new());
		}

		let res = tx.scan(self.beg.clone()..self.end.clone(), INDEX_BATCH_SIZE, 0, None).await?;

		if res.is_empty() {
			self.done = true;
			return Ok(Vec::new());
		}

		// Update begin key for next batch
		if let Some((key, _)) = res.last() {
			self.beg.clone_from(key);
			self.beg.push(0x00);
		}

		decode_record_ids(res)
	}
}

/// Iterator for equality lookups on unique indexes.
///
/// Returns at most one record.
pub(crate) struct UniqueEqualIterator {
	/// The key to look up
	key: Option<Key>,
}

impl UniqueEqualIterator {
	/// Create a new unique equality iterator.
	pub(crate) fn new(
		ns: NamespaceId,
		db: DatabaseId,
		ix: &IndexDefinition,
		value: &Value,
	) -> Result<Self> {
		let array = Array::from(vec![value.clone()]);
		let key = Index::new(ns, db, &ix.table_name, ix.index_id, &array, None).encode_key()?;
		Ok(Self {
			key: Some(key),
		})
	}

	/// Fetch the record ID (if any).
	pub async fn next_batch(&mut self, tx: &Transaction) -> Result<Vec<RecordId>> {
		let Some(key) = self.key.take() else {
			return Ok(Vec::new());
		};

		if let Some(val) = tx.get(&key, None).await? {
			let rid: RecordId = revision::from_slice(&val)?;
			Ok(vec![rid])
		} else {
			Ok(Vec::new())
		}
	}
}

/// Compute the begin key for a non-unique index range scan.
fn compute_index_range_beg_key(
	ns: NamespaceId,
	db: DatabaseId,
	ix: &IndexDefinition,
	from: Option<&RangeBound>,
) -> Result<(Key, bool)> {
	if let Some(from) = from {
		let array = Array::from(vec![from.value.clone()]);
		if from.inclusive {
			Ok((Index::prefix_ids_beg(ns, db, &ix.table_name, ix.index_id, &array)?, true))
		} else {
			Ok((Index::prefix_ids_end(ns, db, &ix.table_name, ix.index_id, &array)?, false))
		}
	} else {
		Ok((Index::prefix_beg(ns, db, &ix.table_name, ix.index_id)?, true))
	}
}

/// Compute the end key for a non-unique index range scan.
fn compute_index_range_end_key(
	ns: NamespaceId,
	db: DatabaseId,
	ix: &IndexDefinition,
	to: Option<&RangeBound>,
) -> Result<(Key, bool)> {
	if let Some(to) = to {
		let array = Array::from(vec![to.value.clone()]);
		if to.inclusive {
			Ok((Index::prefix_ids_end(ns, db, &ix.table_name, ix.index_id, &array)?, true))
		} else {
			Ok((Index::prefix_ids_beg(ns, db, &ix.table_name, ix.index_id, &array)?, false))
		}
	} else {
		Ok((Index::prefix_end(ns, db, &ix.table_name, ix.index_id)?, true))
	}
}

/// Forward iterator for range scans on non-unique indexes.
///
/// Uses `tx.scan()` and advances the `beg` cursor after each batch.
/// An exclusive `beg` boundary is filtered on the first batch only.
pub(crate) struct IndexRangeForwardIterator {
	beg: Key,
	end: Key,
	/// Whether the exclusive `beg` boundary has already been filtered.
	beg_checked: bool,
	done: bool,
}

impl IndexRangeForwardIterator {
	pub(crate) fn new(
		ns: NamespaceId,
		db: DatabaseId,
		ix: &IndexDefinition,
		from: Option<&RangeBound>,
		to: Option<&RangeBound>,
	) -> Result<Self> {
		let (beg, beg_inclusive) = compute_index_range_beg_key(ns, db, ix, from)?;
		let (end, _end_inclusive) = compute_index_range_end_key(ns, db, ix, to)?;

		Ok(Self {
			beg,
			end,
			beg_checked: beg_inclusive,
			done: false,
		})
	}

	pub(crate) async fn next_batch(&mut self, tx: &Transaction) -> Result<Vec<RecordId>> {
		if self.done {
			return Ok(Vec::new());
		}

		// Save the beg key before scan advances it, for exclusive filtering.
		let check_exclusive_beg = if self.beg_checked {
			None
		} else {
			Some(self.beg.clone())
		};

		let res = tx.scan(self.beg.clone()..self.end.clone(), INDEX_BATCH_SIZE, 0, None).await?;

		if res.is_empty() {
			self.done = true;
			return Ok(Vec::new());
		}

		// Advance beg cursor past the last returned key
		if let Some((key, _)) = res.last() {
			self.beg.clone_from(key);
			self.beg.push(0x00);
		}

		self.beg_checked = true;

		let mut records = Vec::with_capacity(res.len());
		for (key, val) in res {
			if let Some(ref exclusive_key) = check_exclusive_beg
				&& key == *exclusive_key
			{
				continue;
			}
			let rid: RecordId = revision::from_slice(&val)?;
			records.push(rid);
		}

		Ok(records)
	}
}

/// Backward iterator for range scans on non-unique indexes.
///
/// Uses `tx.scanr()` and retreats the `end` cursor after each batch.
/// An exclusive `end` boundary is filtered on the first batch.
/// An exclusive `beg` boundary is filtered on every batch (since `beg`
/// is included by the half-open range `[beg, end)`).
pub(crate) struct IndexRangeBackwardIterator {
	beg: Key,
	end: Key,
	/// Whether the exclusive `end` boundary has already been filtered.
	end_checked: bool,
	/// Key to exclude at the `beg` edge (checked on every batch).
	/// Set when the lower bound is exclusive.
	exclude_beg_key: Option<Key>,
	done: bool,
}

impl IndexRangeBackwardIterator {
	pub(crate) fn new(
		ns: NamespaceId,
		db: DatabaseId,
		ix: &IndexDefinition,
		from: Option<&RangeBound>,
		to: Option<&RangeBound>,
	) -> Result<Self> {
		let (beg, beg_inclusive) = compute_index_range_beg_key(ns, db, ix, from)?;
		let (end, end_inclusive) = compute_index_range_end_key(ns, db, ix, to)?;

		let exclude_beg_key = if beg_inclusive {
			None
		} else {
			Some(beg.clone())
		};

		Ok(Self {
			beg,
			end,
			end_checked: end_inclusive,
			exclude_beg_key,
			done: false,
		})
	}

	pub(crate) async fn next_batch(&mut self, tx: &Transaction) -> Result<Vec<RecordId>> {
		if self.done {
			return Ok(Vec::new());
		}

		// Save the end key before scan retreats it, for exclusive filtering.
		let check_exclusive_end = if self.end_checked {
			None
		} else {
			Some(self.end.clone())
		};

		let res = tx.scanr(self.beg.clone()..self.end.clone(), INDEX_BATCH_SIZE, 0, None).await?;

		if res.is_empty() {
			self.done = true;
			return Ok(Vec::new());
		}

		// Retreat end cursor to the last returned key
		if let Some((key, _)) = res.last() {
			self.end.clone_from(key);
		}

		self.end_checked = true;

		let mut records = Vec::with_capacity(res.len());
		for (key, val) in res {
			if let Some(ref exclusive_key) = check_exclusive_end
				&& key == *exclusive_key
			{
				continue;
			}
			if let Some(ref beg_key) = self.exclude_beg_key
				&& key == *beg_key
			{
				continue;
			}
			let rid: RecordId = revision::from_slice(&val)?;
			records.push(rid);
		}

		Ok(records)
	}
}

/// Enum dispatching range scans on non-unique indexes to the
/// appropriate direction-specific iterator.
pub(crate) enum IndexRangeIterator {
	Forward(IndexRangeForwardIterator),
	Backward(IndexRangeBackwardIterator),
}

impl IndexRangeIterator {
	pub(crate) fn new(
		ns: NamespaceId,
		db: DatabaseId,
		ix: &IndexDefinition,
		from: Option<&RangeBound>,
		to: Option<&RangeBound>,
		direction: ScanDirection,
	) -> Result<Self> {
		match direction {
			ScanDirection::Forward => {
				Ok(Self::Forward(IndexRangeForwardIterator::new(ns, db, ix, from, to)?))
			}
			ScanDirection::Backward => {
				Ok(Self::Backward(IndexRangeBackwardIterator::new(ns, db, ix, from, to)?))
			}
		}
	}

	pub(crate) async fn next_batch(&mut self, tx: &Transaction) -> Result<Vec<RecordId>> {
		match self {
			Self::Forward(iter) => iter.next_batch(tx).await,
			Self::Backward(iter) => iter.next_batch(tx).await,
		}
	}
}

// ---------------------------------------------------------------------------
// Unique-index range helpers
// ---------------------------------------------------------------------------

/// Compute the begin key for a unique index range scan.
fn compute_unique_range_beg_key(
	ns: NamespaceId,
	db: DatabaseId,
	ix: &IndexDefinition,
	from: Option<&RangeBound>,
) -> Result<(Key, bool)> {
	if let Some(from) = from {
		let array = Array::from(vec![from.value.clone()]);
		let key = Index::new(ns, db, &ix.table_name, ix.index_id, &array, None).encode_key()?;
		Ok((key, from.inclusive))
	} else {
		Ok((Index::prefix_beg(ns, db, &ix.table_name, ix.index_id)?, true))
	}
}

/// Compute the end key for a unique index range scan.
fn compute_unique_range_end_key(
	ns: NamespaceId,
	db: DatabaseId,
	ix: &IndexDefinition,
	to: Option<&RangeBound>,
) -> Result<(Key, bool)> {
	if let Some(to) = to {
		let array = Array::from(vec![to.value.clone()]);
		let key = Index::new(ns, db, &ix.table_name, ix.index_id, &array, None).encode_key()?;
		Ok((key, to.inclusive))
	} else {
		Ok((Index::prefix_end(ns, db, &ix.table_name, ix.index_id)?, true))
	}
}

/// Forward iterator for range scans on unique indexes.
///
/// Uses `tx.scan()` and advances the `beg` cursor after each batch.
/// An exclusive `beg` boundary is filtered on the first batch.
/// An inclusive `end` boundary triggers a final `get()` when the scan
/// is exhausted (because the half-open range `[beg, end)` excludes `end`).
pub(crate) struct UniqueRangeForwardIterator {
	beg: Key,
	end: Key,
	/// Whether the exclusive `beg` boundary has already been filtered.
	beg_checked: bool,
	/// Whether an inclusive `end` needs a trailing get().
	end_inclusive: bool,
	done: bool,
}

impl UniqueRangeForwardIterator {
	pub(crate) fn new(
		ns: NamespaceId,
		db: DatabaseId,
		ix: &IndexDefinition,
		from: Option<&RangeBound>,
		to: Option<&RangeBound>,
	) -> Result<Self> {
		let (beg, beg_inclusive) = compute_unique_range_beg_key(ns, db, ix, from)?;
		let (end, end_inclusive) = compute_unique_range_end_key(ns, db, ix, to)?;

		Ok(Self {
			beg,
			end,
			beg_checked: beg_inclusive,
			end_inclusive,
			done: false,
		})
	}

	pub(crate) async fn next_batch(&mut self, tx: &Transaction) -> Result<Vec<RecordId>> {
		if self.done {
			return Ok(Vec::new());
		}

		let check_exclusive_beg = if self.beg_checked {
			None
		} else {
			Some(self.beg.clone())
		};

		let limit = INDEX_BATCH_SIZE + 1;
		let res = tx.scan(self.beg.clone()..self.end.clone(), limit, 0, None).await?;

		if res.is_empty() {
			self.done = true;
			if self.end_inclusive
				&& let Some(val) = tx.get(&self.end, None).await?
			{
				let rid: RecordId = revision::from_slice(&val)?;
				return Ok(vec![rid]);
			}
			return Ok(Vec::new());
		}

		if let Some((key, _)) = res.last() {
			self.beg.clone_from(key);
			self.beg.push(0x00);
		}

		self.beg_checked = true;

		let mut records = Vec::with_capacity(res.len());
		for (key, val) in res {
			if let Some(ref exclusive_key) = check_exclusive_beg
				&& key == *exclusive_key
			{
				continue;
			}
			let rid: RecordId = revision::from_slice(&val)?;
			records.push(rid);
		}

		Ok(records)
	}
}

/// Backward iterator for range scans on unique indexes.
///
/// Uses `tx.scanr()` and retreats the `end` cursor after each batch.
/// An exclusive `end` boundary is filtered on the first batch.
/// An exclusive `beg` boundary is filtered on every batch (since `beg`
/// is included by the half-open range `[beg, end)`).
pub(crate) struct UniqueRangeBackwardIterator {
	beg: Key,
	end: Key,
	/// Whether the exclusive `end` boundary has already been filtered.
	end_checked: bool,
	/// Key to exclude at the `beg` edge (checked on every batch).
	exclude_beg_key: Option<Key>,
	done: bool,
}

impl UniqueRangeBackwardIterator {
	pub(crate) fn new(
		ns: NamespaceId,
		db: DatabaseId,
		ix: &IndexDefinition,
		from: Option<&RangeBound>,
		to: Option<&RangeBound>,
	) -> Result<Self> {
		let (beg, beg_inclusive) = compute_unique_range_beg_key(ns, db, ix, from)?;
		let (end, end_inclusive) = compute_unique_range_end_key(ns, db, ix, to)?;

		let exclude_beg_key = if beg_inclusive {
			None
		} else {
			Some(beg.clone())
		};

		Ok(Self {
			beg,
			end,
			end_checked: end_inclusive,
			exclude_beg_key,
			done: false,
		})
	}

	pub(crate) async fn next_batch(&mut self, tx: &Transaction) -> Result<Vec<RecordId>> {
		if self.done {
			return Ok(Vec::new());
		}

		let check_exclusive_end = if self.end_checked {
			None
		} else {
			Some(self.end.clone())
		};

		let limit = INDEX_BATCH_SIZE + 1;
		let res = tx.scanr(self.beg.clone()..self.end.clone(), limit, 0, None).await?;

		if res.is_empty() {
			self.done = true;
			return Ok(Vec::new());
		}

		if let Some((key, _)) = res.last() {
			self.end.clone_from(key);
		}

		self.end_checked = true;

		let mut records = Vec::with_capacity(res.len());
		for (key, val) in res {
			if let Some(ref exclusive_key) = check_exclusive_end
				&& key == *exclusive_key
			{
				continue;
			}
			if let Some(ref beg_key) = self.exclude_beg_key
				&& key == *beg_key
			{
				continue;
			}
			let rid: RecordId = revision::from_slice(&val)?;
			records.push(rid);
		}

		Ok(records)
	}
}

/// Enum dispatching range scans on unique indexes to the
/// appropriate direction-specific iterator.
pub(crate) enum UniqueRangeIterator {
	Forward(UniqueRangeForwardIterator),
	Backward(UniqueRangeBackwardIterator),
}

impl UniqueRangeIterator {
	pub(crate) fn new(
		ns: NamespaceId,
		db: DatabaseId,
		ix: &IndexDefinition,
		from: Option<&RangeBound>,
		to: Option<&RangeBound>,
		direction: ScanDirection,
	) -> Result<Self> {
		match direction {
			ScanDirection::Forward => {
				Ok(Self::Forward(UniqueRangeForwardIterator::new(ns, db, ix, from, to)?))
			}
			ScanDirection::Backward => {
				Ok(Self::Backward(UniqueRangeBackwardIterator::new(ns, db, ix, from, to)?))
			}
		}
	}

	pub(crate) async fn next_batch(&mut self, tx: &Transaction) -> Result<Vec<RecordId>> {
		match self {
			Self::Forward(iter) => iter.next_batch(tx).await,
			Self::Backward(iter) => iter.next_batch(tx).await,
		}
	}
}

/// Iterator for compound (multi-column) index equality scans.
///
/// Supports both forward and backward scanning, controlled by [`ScanDirection`].
/// Forward scans use `tx.scan()` and advance the `beg` cursor;
/// backward scans use `tx.scanr()` and retreat the `end` cursor.
pub(crate) struct CompoundEqualIterator {
	/// Current scan position (begin key)
	beg: Vec<u8>,
	/// End key (exclusive)
	end: Vec<u8>,
	/// Whether iteration is complete
	done: bool,
	/// Scan direction
	direction: ScanDirection,
}

impl CompoundEqualIterator {
	/// Create a new compound equality iterator.
	///
	/// `prefix` contains the fixed equality values for leading columns.
	/// When an additional equality range is present, it is appended to the
	/// prefix so the scan covers the exact composite key.
	pub(crate) fn new(
		ns: NamespaceId,
		db: DatabaseId,
		ix: &IndexDefinition,
		prefix: &[Value],
		range: Option<&(BinaryOperator, Value)>,
		direction: ScanDirection,
	) -> Result<Self> {
		let (beg, end) = compute_compound_key_range(ns, db, ix, prefix, range)?;
		Ok(Self {
			beg,
			end,
			done: false,
			direction,
		})
	}

	/// Fetch the next batch of record IDs, capped at `limit`.
	///
	/// The caller supplies a `limit` so that storage-level scans can be
	/// bounded (e.g. when a pushed-down LIMIT is active).  Pass
	/// `INDEX_BATCH_SIZE` when no external limit applies.
	pub(crate) async fn next_batch(
		&mut self,
		tx: &Transaction,
		limit: u32,
	) -> Result<Vec<RecordId>> {
		if self.done {
			return Ok(Vec::new());
		}

		let scan_limit = limit.min(INDEX_BATCH_SIZE);
		let res = match self.direction {
			ScanDirection::Forward => {
				tx.scan(self.beg.clone()..self.end.clone(), scan_limit, 0, None).await?
			}
			ScanDirection::Backward => {
				tx.scanr(self.beg.clone()..self.end.clone(), scan_limit, 0, None).await?
			}
		};

		if res.is_empty() {
			self.done = true;
			return Ok(Vec::new());
		}

		// Update cursor for next batch
		if let Some((key, _)) = res.last() {
			match self.direction {
				ScanDirection::Forward => {
					self.beg.clone_from(key);
					self.beg.push(0x00);
				}
				ScanDirection::Backward => {
					self.end.clone_from(key);
				}
			}
		}

		decode_record_ids(res)
	}
}

/// Iterator for compound (multi-column) index range scans.
///
/// Handles the case where leading columns are fixed by equality and the
/// next column has a range condition (e.g. `a = 1 AND b > 5`).
pub(crate) struct CompoundRangeForwardIterator {
	/// Current scan position (begin key)
	beg: Vec<u8>,
	/// End key (exclusive)
	end: Vec<u8>,
	/// Whether iteration is complete
	done: bool,
}

impl CompoundRangeForwardIterator {
	/// Create a new compound range iterator.
	pub(crate) fn new(
		ns: NamespaceId,
		db: DatabaseId,
		ix: &IndexDefinition,
		prefix: &[Value],
		range: &(BinaryOperator, Value),
	) -> Result<Self> {
		let (beg, end) = compute_compound_key_range(ns, db, ix, prefix, Some(range))?;
		Ok(Self {
			beg,
			end,
			done: false,
		})
	}

	/// Fetch the next batch of record IDs, capped at `limit`.
	pub(crate) async fn next_batch(
		&mut self,
		tx: &Transaction,
		limit: u32,
	) -> Result<Vec<RecordId>> {
		if self.done {
			return Ok(Vec::new());
		}

		let scan_limit = limit.min(INDEX_BATCH_SIZE);
		let res = tx.scan(self.beg.clone()..self.end.clone(), scan_limit, 0, None).await?;

		if res.is_empty() {
			self.done = true;
			return Ok(Vec::new());
		}

		// Advance cursor past the last key for the next batch
		if let Some((key, _)) = res.last() {
			self.beg.clone_from(key);
			self.beg.push(0x00);
		}

		decode_record_ids(res)
	}
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute the KV key range `(beg, end)` for a compound index scan.
///
/// Builds the appropriate prefix-based key boundaries depending on whether
/// the scan is a pure equality prefix or has a range condition on the
/// next column.
fn compute_compound_key_range(
	ns: NamespaceId,
	db: DatabaseId,
	ix: &IndexDefinition,
	prefix: &[Value],
	range: Option<&(BinaryOperator, Value)>,
) -> Result<(Vec<u8>, Vec<u8>)> {
	let prefix_array = Array::from(prefix.to_vec());

	if let Some((op, val)) = range {
		let mut key_values: Vec<Value> = prefix.to_vec();
		key_values.push(val.clone());
		let key_array = Array::from(key_values);

		match op {
			BinaryOperator::Equal | BinaryOperator::ExactEqual => {
				let beg = Index::prefix_ids_composite_beg(
					ns,
					db,
					&ix.table_name,
					ix.index_id,
					&key_array,
				)?;
				let end = Index::prefix_ids_composite_end(
					ns,
					db,
					&ix.table_name,
					ix.index_id,
					&key_array,
				)?;
				Ok((beg, end))
			}
			BinaryOperator::MoreThan => {
				let beg = Index::prefix_ids_end(ns, db, &ix.table_name, ix.index_id, &key_array)?;
				let end = Index::prefix_ids_composite_end(
					ns,
					db,
					&ix.table_name,
					ix.index_id,
					&prefix_array,
				)?;
				Ok((beg, end))
			}
			BinaryOperator::MoreThanEqual => {
				let beg = Index::prefix_ids_beg(ns, db, &ix.table_name, ix.index_id, &key_array)?;
				let end = Index::prefix_ids_composite_end(
					ns,
					db,
					&ix.table_name,
					ix.index_id,
					&prefix_array,
				)?;
				Ok((beg, end))
			}
			BinaryOperator::LessThan => {
				let beg = Index::prefix_ids_composite_beg(
					ns,
					db,
					&ix.table_name,
					ix.index_id,
					&prefix_array,
				)?;
				let end = Index::prefix_ids_beg(ns, db, &ix.table_name, ix.index_id, &key_array)?;
				Ok((beg, end))
			}
			BinaryOperator::LessThanEqual => {
				let beg = Index::prefix_ids_composite_beg(
					ns,
					db,
					&ix.table_name,
					ix.index_id,
					&prefix_array,
				)?;
				let end = Index::prefix_ids_end(ns, db, &ix.table_name, ix.index_id, &key_array)?;
				Ok((beg, end))
			}
			_ => {
				let beg = Index::prefix_ids_composite_beg(
					ns,
					db,
					&ix.table_name,
					ix.index_id,
					&prefix_array,
				)?;
				let end = Index::prefix_ids_composite_end(
					ns,
					db,
					&ix.table_name,
					ix.index_id,
					&prefix_array,
				)?;
				Ok((beg, end))
			}
		}
	} else {
		let beg =
			Index::prefix_ids_composite_beg(ns, db, &ix.table_name, ix.index_id, &prefix_array)?;
		let end =
			Index::prefix_ids_composite_end(ns, db, &ix.table_name, ix.index_id, &prefix_array)?;
		Ok((beg, end))
	}
}
