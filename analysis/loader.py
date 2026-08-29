"""Safely load an uploaded NPZ game and route it to a schema adapter."""

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

from analysis.errors import AnalysisFileError
from analysis.schema_v1 import adapt_schema_v1


MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 64


def _inspect_npz_archive(file_bytes):
    """Reject malformed or unexpectedly large ZIP containers before NumPy runs."""
    try:
        with ZipFile(BytesIO(file_bytes)) as archive:
            members = archive.infolist()
            if not members:
                raise AnalysisFileError("The selected NPZ archive is empty")
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise AnalysisFileError(
                    f"The NPZ archive contains more than {MAX_ARCHIVE_MEMBERS} arrays"
                )

            uncompressed_size = 0
            for member in members:
                if member.flag_bits & 0x1:
                    raise AnalysisFileError("Encrypted NPZ archives are not supported")
                if not member.filename.endswith(".npy"):
                    raise AnalysisFileError(
                        "The selected archive contains a non-NumPy member"
                    )
                uncompressed_size += member.file_size

            if uncompressed_size > MAX_UNCOMPRESSED_BYTES:
                raise AnalysisFileError(
                    "The uncompressed NPZ data is larger than the analyzer limit"
                )
    except BadZipFile as error:
        raise AnalysisFileError("The selected file is not a valid NPZ archive") from error


def load_analysis_bytes(file_bytes, file_name="game.npz"):
    """Load one in-memory NPZ file into the canonical analysis model."""
    if not file_bytes:
        raise AnalysisFileError("The selected file is empty")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise AnalysisFileError(
            f"The selected file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
        )
    if Path(file_name).suffix.lower() != ".npz":
        raise AnalysisFileError("Choose a file with the .npz extension")

    _inspect_npz_archive(file_bytes)

    try:
        with np.load(BytesIO(file_bytes), allow_pickle=False) as stored:
            if "schema_version" not in stored.files:
                raise AnalysisFileError("The NPZ file does not declare a schema version")
            arrays = {name: stored[name].copy() for name in stored.files}
    except AnalysisFileError:
        raise
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise AnalysisFileError(
            "NumPy could not read the selected NPZ game safely"
        ) from error

    schema_version_array = arrays["schema_version"]
    if schema_version_array.shape != ():
        raise AnalysisFileError("'schema_version' must be one scalar value")
    try:
        schema_version = int(schema_version_array.item())
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisFileError("'schema_version' must be an integer") from error

    adapters = {
        1: adapt_schema_v1,
    }
    adapter = adapters.get(schema_version)
    if adapter is None:
        supported_versions = ", ".join(str(version) for version in adapters)
        raise AnalysisFileError(
            f"Unsupported NPZ schema version {schema_version}. "
            f"Supported versions: {supported_versions}"
        )

    try:
        return adapter(arrays, file_name)
    except AnalysisFileError:
        raise
    except (TypeError, ValueError, OverflowError, IndexError) as error:
        # Storage files are an external input boundary. Conversion failures
        # should become a useful validation message rather than an HTTP 500.
        raise AnalysisFileError(
            f"Schema version {schema_version} contains a field with the wrong type"
        ) from error


def load_analysis_path(path):
    """Convenience loader used by tests and future command-line workflows."""
    file_path = Path(path)
    try:
        file_bytes = file_path.read_bytes()
    except OSError as error:
        raise AnalysisFileError(f"Could not read '{file_path.name}'") from error
    return load_analysis_bytes(file_bytes, file_path.name)
