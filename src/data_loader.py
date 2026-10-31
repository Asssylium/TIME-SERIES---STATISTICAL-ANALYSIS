"""Read and validate daily draws, preserving within-day duplicates.

CSV supports ``date,draw_01,...,draw_27`` or ``date,draws``. In the latter
format, ``draws`` is a quoted JSON array or a comma/space/semicolon/pipe
separated string. JSON must be a list of objects with ``date`` and ``draws``
keys, where ``draws`` is a list. Dates must use YYYY-MM-DD.

Canonical frames contain a datetime64 date column and 27 uint8 draw columns,
sorted by date. Missing calendar dates are preserved as gaps, never imputed.
The original DataLoader reserves column 0 and accepts only draws 01..99.
XSMBDataLoader accepts real XSMB outcomes 00..99 and returns date-indexed
DataFrames from either frequency CSVs or raw prize columns. Binary matrices
represent presence (multi-hot), not mutually exclusive classes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
from numpy.typing import NDArray


PathLike = str | Path
DRAW_COUNT = 27
MATRIX_WIDTH = 100
DRAW_COLUMNS = tuple(f"draw_{i:02d}" for i in range(1, DRAW_COUNT + 1))
NUMBER_COLUMNS = tuple(f"{number:02d}" for number in range(MATRIX_WIDTH))
DEFAULT_XSMB_URL = (
    "https://raw.githubusercontent.com/khiemdoan/"
    "vietnam-lottery-xsmb-analysis/refs/heads/main/data/xsmb-sparse.csv"
)


class DataValidationError(ValueError):
    """Raised when daily records violate the input contract."""


class XSMBDataLoader:
    """Load XSMB CSV data and build frequency/presence matrices for 00..99.

    The default URL already contains daily frequencies, not raw prize values.
    A complete set of columns '00'..'99' selects frequency mode. Otherwise,
    prize columns contain individual integers or strings separated by spaces,
    commas, semicolons, or pipes, e.g. '12345 67890'. The last two digits of
    each prize are retained. Integer prizes 0..9 are padded to '00'..'09'.

    Missing values, duplicate dates, malformed values, and daily totals other
    than 27 raise DataValidationError. Missing calendar dates are not filled;
    downstream features must use the date index to handle gaps explicitly.
    """

    def __init__(
        self,
        url: str = DEFAULT_XSMB_URL,
        *,
        date_column: str = "date",
        date_format: str = "%Y-%m-%d",
        prize_columns: Sequence[str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Configure the source, date format, and optional raw prize columns.

        Set prize_columns when a raw CSV also contains metadata columns.
        Otherwise every non-date column is treated as a prize column unless
        the frequency schema is detected. An explicit prize_columns list
        selects raw prize mode even when column names are two-digit numbers.
        """
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive.")
        if prize_columns is not None:
            if isinstance(prize_columns, (str, bytes)):
                raise ValueError("prize_columns must be a sequence of column names.")
            if not prize_columns or len(set(prize_columns)) != len(prize_columns):
                raise ValueError("prize_columns must be nonempty and unique.")
            if date_column in prize_columns:
                raise ValueError("The date column cannot also be a prize column.")
        self.url = url
        self.date_column = date_column
        self.date_format = date_format
        self.prize_columns = tuple(prize_columns) if prize_columns is not None else None
        self.timeout = timeout
        self.raw_data: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        """Download, cache, and return raw CSV data with strings preserved.

        No date sorting or numeric conversion occurs here. Network/IO errors
        propagate; empty or malformed CSV content raises DataValidationError.
        Returned data is a copy so caller edits do not change the cached input.
        """
        with urlopen(self.url, timeout=self.timeout) as response:
            content = BytesIO(response.read())
        try:
            frame = pd.read_csv(
                content, dtype="string", keep_default_na=False, encoding="utf-8-sig"
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise DataValidationError(f"Invalid XSMB CSV: {exc}") from exc
        if frame.empty:
            raise DataValidationError("XSMB CSV contains no daily records.")
        self.raw_data = frame
        return frame.copy(deep=True)

    def inspect_data(self, data: pd.DataFrame | None = None, n: int = 5) -> None:
        """Print info() and the first n rows; load the source if not yet cached."""
        frame = self._get_data(data)
        frame.info()
        print(frame.head(n))

    def preprocess_and_build_matrices(
        self, data: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (frequency_matrix, binary_matrix), both shaped (N_days, 100).

        Each matrix has a sorted, unique DatetimeIndex named after date_column,
        columns '00'..'99', and uint8 values. Frequency rows sum to exactly 27;
        binary entries equal (frequency > 0). '00' is a valid XSMB outcome.
        Input data is never modified. Omit data to use the cached/downloaded CSV.
        """
        frame = self._get_data(data)
        if frame.empty or not frame.columns.is_unique:
            raise DataValidationError("Expected nonempty data with unique column names.")
        if self.date_column not in frame.columns:
            raise DataValidationError(f"Missing date column: {self.date_column!r}")
        dates = self._parse_dates(frame[self.date_column])
        numbered_columns = set(NUMBER_COLUMNS).intersection(frame.columns)
        if self.prize_columns is None and numbered_columns:
            missing = set(NUMBER_COLUMNS) - numbered_columns
            if missing:
                raise DataValidationError(
                    f"Incomplete frequency schema; missing columns: {sorted(missing)}. "
                    "For raw prizes, specify prize_columns explicitly."
                )
            extra = set(frame.columns) - set(NUMBER_COLUMNS) - {self.date_column}
            if extra:
                raise DataValidationError(f"Unexpected frequency columns: {sorted(extra)}")
            counts = self._read_frequencies(frame)
        else:
            counts = self._count_prizes(frame)

        totals = counts.sum(axis=1, dtype=np.int64)
        invalid = totals != DRAW_COUNT
        if invalid.any():
            row = int(np.flatnonzero(invalid)[0])
            raise DataValidationError(
                f"Date {dates[row]:%Y-%m-%d}: expected 27 results, got {totals[row]}."
            )
        frequency = pd.DataFrame(
            counts.astype(np.uint8), index=dates, columns=NUMBER_COLUMNS
        ).sort_index()
        binary = frequency.gt(0).astype(np.uint8)
        return frequency, binary

    def _get_data(self, data: pd.DataFrame | None) -> pd.DataFrame:
        """Resolve explicit data, cached data, or a fresh download."""
        if data is not None:
            if not isinstance(data, pd.DataFrame):
                raise TypeError("data must be a pandas DataFrame.")
            return data
        if self.raw_data is None:
            self.load_data()
        assert self.raw_data is not None
        return self.raw_data

    def _parse_dates(self, values: pd.Series) -> pd.DatetimeIndex:
        """Parse dates strictly, rejecting missing, timed, and duplicate entries."""
        if pd.api.types.is_datetime64_any_dtype(values.dtype):
            parsed = values
        else:
            parsed = pd.to_datetime(
                values.astype("string").str.strip(),
                format=self.date_format,
                errors="coerce",
            )
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise DataValidationError("Dates must be timezone-naive daily dates.")
        invalid = parsed.isna() | parsed.ne(parsed.dt.normalize())
        if invalid.any():
            row = int(np.flatnonzero(invalid.to_numpy())[0])
            raise DataValidationError(f"Record {row + 1}: invalid daily date {values.iloc[row]!r}.")
        if parsed.duplicated().any():
            duplicate = parsed[parsed.duplicated()].iloc[0]
            raise DataValidationError(f"Duplicate date: {duplicate:%Y-%m-%d}")
        return pd.DatetimeIndex(parsed, name=self.date_column)

    @staticmethod
    def _read_frequencies(frame: pd.DataFrame) -> NDArray[np.int64]:
        """Read existing counts without interpreting them as prize results."""
        flat = frame.loc[:, list(NUMBER_COLUMNS)].to_numpy().ravel()
        text = pd.Series(flat, dtype="string").str.strip()
        # Validate before uint8 conversion to avoid truncation or wraparound.
        valid = text.str.fullmatch(r"(?:[0-9]|[01][0-9]|2[0-7])").fillna(False)
        if not valid.all():
            position = int(np.flatnonzero(~valid.to_numpy(dtype=bool))[0])
            row, column = divmod(position, MATRIX_WIDTH)
            raise DataValidationError(
                f"Record {row + 1}, column {NUMBER_COLUMNS[column]}: "
                f"expected an integer count 0..27, got {flat[position]!r}."
            )
        return text.to_numpy(dtype=np.int64).reshape(len(frame), MATRIX_WIDTH)

    def _count_prizes(self, frame: pd.DataFrame) -> NDArray[np.int64]:
        """Tokenize prize cells and count two-digit endings using bincount."""
        columns = (
            list(self.prize_columns) if self.prize_columns is not None
            else [column for column in frame.columns if column != self.date_column]
        )
        if not columns:
            raise DataValidationError("At least one prize column is required.")
        missing = set(columns) - set(frame.columns)
        if missing:
            raise DataValidationError(f"Missing prize columns: {sorted(missing)}")
        flat = frame.loc[:, columns].to_numpy().ravel()
        cells = pd.Series(flat, dtype="string").str.strip()
        separator = r"(?:\s*[,;|]\s*|\s+)"
        valid = cells.str.fullmatch(
            rf"[0-9]{{1,5}}(?:{separator}[0-9]{{1,5}})*"
        ).fillna(False)
        if not valid.all():
            position = int(np.flatnonzero(~valid.to_numpy(dtype=bool))[0])
            row, column = divmod(position, len(columns))
            raise DataValidationError(
                f"Record {row + 1}, column {columns[column]!r}: "
                f"invalid prize value {flat[position]!r}; expected 1..5 digit integers "
                "separated by spaces, commas, semicolons, or pipes."
            )
        tokens = cells.str.split(separator, regex=True).explode()
        endings = tokens.str[-2:].str.zfill(2).to_numpy(dtype=np.int64)
        row_ids = tokens.index.to_numpy(dtype=np.int64) // len(columns)
        counts = np.bincount(
            row_ids * MATRIX_WIDTH + endings, minlength=len(frame) * MATRIX_WIDTH
        )
        return counts.reshape(len(frame), MATRIX_WIDTH)


class DataLoader:
    """Load daily draws into validated, chronological pandas frames.

    Input values must be integers 1..99 or digit strings such as '1'/'01'.
    Booleans, floats, missing values, zero and out-of-range values are rejected.
    Duplicate dates are rejected; duplicate draws within a date are retained.
    IO errors propagate unchanged; invalid file contents raise
    :class:`DataValidationError`.
    """

    def load_csv(self, path: PathLike) -> pd.DataFrame:
        """Read UTF-8 CSV using either of the schemas in the module docstring.

        Reading as strings prevents pandas from silently coercing draw values.
        Only the documented columns are accepted; mixed schemas are rejected.
        """
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False,
                                encoding="utf-8-sig")
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise DataValidationError(f"Invalid CSV: {exc}") from exc
        if "date" not in frame.columns:
            raise DataValidationError("CSV requires a 'date' column.")
        if "draws" in frame.columns:
            if set(frame.columns) != {"date", "draws"}:
                raise DataValidationError("Packed CSV requires exactly 'date' and 'draws'.")
            records = [
                {"date": date, "draws": self._parse_draws(value, row)}
                for row, (date, value) in enumerate(
                    zip(frame["date"], frame["draws"]), start=1
                )
            ]
            return self.from_records(records)
        return self.validate(frame)

    def load_json(self, path: PathLike) -> pd.DataFrame:
        """Read a JSON array of {"date": "YYYY-MM-DD", "draws": [...]} records."""
        try:
            with Path(path).open(encoding="utf-8-sig") as handle:
                records = json.load(handle)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"Invalid JSON: {exc}") from exc
        if not isinstance(records, list):
            raise DataValidationError("JSON root must be a list of daily records.")
        return self.from_records(records)

    def from_records(self, records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
        """Validate in-memory records using the same schema as JSON input."""
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise DataValidationError("Records must be a sequence of mappings.")
        if len(records) == 0:
            raise DataValidationError("At least one daily record is required.")
        dates: list[Any] = []
        daily_draws: list[list[Any]] = []
        for row, record in enumerate(records, start=1):
            if not isinstance(record, Mapping) or not {"date", "draws"} <= record.keys():
                raise DataValidationError(f"Record {row} requires 'date' and 'draws'.")
            draws = record["draws"]
            if not isinstance(draws, (list, tuple, np.ndarray)):
                raise DataValidationError(f"Record {row}: 'draws' must be a list.")
            if isinstance(draws, np.ndarray) and draws.ndim != 1:
                raise DataValidationError(f"Record {row}: 'draws' must be one-dimensional.")
            if len(draws) != DRAW_COUNT:
                raise DataValidationError(
                    f"Record {row}: expected {DRAW_COUNT} draws, got {len(draws)}."
                )
            dates.append(record["date"])
            daily_draws.append(list(draws))
        # Object dtype preserves invalid booleans/floats until strict validation.
        frame = pd.DataFrame(daily_draws, columns=DRAW_COLUMNS, dtype=object)
        frame.insert(0, "date", dates)
        return self.validate(frame)

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a clean copy; reject invalid values, dates, or missing columns.

        String dates require YYYY-MM-DD. Timezone-naive datetime64 dates are
        also accepted so that already loaded frames can be validated again.
        Timestamps with times of day and duplicate calendar dates are rejected.
        """
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise DataValidationError("Expected a nonempty pandas DataFrame.")
        if not frame.columns.is_unique:
            raise DataValidationError("Duplicate column names are not allowed.")
        required = ["date", *DRAW_COLUMNS]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {', '.join(missing)}")
        extra = [column for column in frame.columns if column not in required]
        if extra:
            raise DataValidationError(f"Unexpected columns: {extra}")
        clean = frame.loc[:, required].copy().reset_index(drop=True)
        dates = clean["date"]
        if pd.api.types.is_datetime64_any_dtype(dates.dtype):
            if isinstance(dates.dtype, pd.DatetimeTZDtype):
                raise DataValidationError("Dates must not have a timezone.")
            parsed_dates = dates
        else:
            date_text = dates.astype("string")
            valid_format = date_text.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}").fillna(False)
            if not valid_format.all():
                bad = int(np.flatnonzero(~valid_format.to_numpy(dtype=bool))[0]) + 1
                raise DataValidationError(f"Record {bad}: date must use YYYY-MM-DD.")
            parsed_dates = pd.to_datetime(date_text, format="%Y-%m-%d", errors="coerce")
        invalid_dates = parsed_dates.isna() | parsed_dates.ne(parsed_dates.dt.normalize())
        if invalid_dates.any():
            bad = int(np.flatnonzero(invalid_dates.to_numpy())[0]) + 1
            raise DataValidationError(f"Record {bad}: invalid date or non-midnight timestamp.")
        if parsed_dates.duplicated().any():
            duplicate = parsed_dates[parsed_dates.duplicated()].iloc[0]
            raise DataValidationError(f"Duplicate date: {duplicate:%Y-%m-%d}")

        raw = clean.loc[:, list(DRAW_COLUMNS)].to_numpy(dtype=object)
        flat = raw.ravel()
        valid_types = np.fromiter(
            (isinstance(value, (str, int, np.integer))
             and not isinstance(value, (bool, np.bool_)) for value in flat),
            dtype=bool, count=flat.size,
        )
        tokens = pd.Series(flat, dtype="string").str.strip()
        valid_tokens = tokens.str.fullmatch(r"(?:0?[1-9]|[1-9][0-9])").fillna(False)
        valid = valid_types & valid_tokens.to_numpy(dtype=bool)
        if not valid.all():
            position = int(np.flatnonzero(~valid)[0])
            row, column = divmod(position, DRAW_COUNT)
            raise DataValidationError(
                f"Record {row + 1}, {DRAW_COLUMNS[column]}: expected integer 01..99, "
                f"got {flat[position]!r}."
            )
        values = tokens.to_numpy(dtype=np.uint8).reshape(raw.shape)
        result = pd.DataFrame(values, columns=DRAW_COLUMNS)
        result.insert(0, "date", parsed_dates.to_numpy())
        return result.sort_values("date", kind="stable").reset_index(drop=True)

    def to_matrices(
        self, frame: pd.DataFrame
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Return (frequency, binary), each shaped (N_days, 100), in date order.

        A repeated draw contributes its full count to frequency and exactly one
        to presence. Each frequency row sums to 27; column 00 is always zero.
        Vectorized bincount avoids Python loops over days or matrix cells.
        """
        clean = self.validate(frame)
        draws = clean.loc[:, list(DRAW_COLUMNS)].to_numpy(dtype=np.int64)
        indices = draws + np.arange(len(clean), dtype=np.int64)[:, None] * MATRIX_WIDTH
        frequency = np.bincount(indices.ravel(), minlength=len(clean) * MATRIX_WIDTH)
        frequency = frequency.reshape(len(clean), MATRIX_WIDTH).astype(np.uint8)
        binary = (frequency > 0).astype(np.uint8)
        return frequency, binary

    def to_padded_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a validated copy with draw values formatted as '01'..'99'."""
        clean = self.validate(frame)
        draws = clean.loc[:, list(DRAW_COLUMNS)].to_numpy()
        padded = pd.DataFrame(np.char.mod("%02d", draws), columns=DRAW_COLUMNS)
        padded.insert(0, "date", clean["date"].to_numpy())
        return padded

    @staticmethod
    def _parse_draws(value: str, row: int) -> list[Any]:
        """Decode one packed CSV cell; final value/count checks happen later."""
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"Record {row}: invalid draws JSON.") from exc
            if not isinstance(parsed, list):
                raise DataValidationError(f"Record {row}: expected a JSON list.")
            return parsed
        # Empty entries between explicit delimiters stay empty and fail validation.
        return re.split(r"\s*[,;|]\s*|\s+", text)
