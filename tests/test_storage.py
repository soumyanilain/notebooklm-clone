import pytest

from notebooklm.storage import LocalStorage, NotebookStore, StorageError, safe_filename


def test_notebook_crud(tmp_path):
    store = NotebookStore(LocalStorage(tmp_path))
    a = store.create_notebook("Biology")
    b = store.create_notebook("History")
    assert {n.name for n in store.list_notebooks()} == {"Biology", "History"}
    with pytest.raises(StorageError):
        store.create_notebook("biology")  # names are unique, case-insensitive
    with pytest.raises(StorageError):
        store.create_notebook("   ")
    store.rename_notebook(a.id, "Cell Biology")
    assert store.get_notebook(a.id).name == "Cell Biology"
    store.delete_notebook(b.id)
    assert [n.id for n in store.list_notebooks()] == [a.id]
    assert not (tmp_path / "notebooks" / b.id).exists()


def test_keys_cannot_escape_data_dir(tmp_path):
    backend = LocalStorage(tmp_path / "data")
    for bad in ("../outside.txt", "/etc/passwd", "notebooks/../../x"):
        with pytest.raises(StorageError):
            backend.write_text(bad, "x")
    with pytest.raises(StorageError):
        NotebookStore(backend).get_notebook("../../etc")


def test_safe_filename():
    assert safe_filename("../../evil name?.pdf") == "evil name_.pdf"
    assert safe_filename("C:\\Users\\me\\notes.txt") == "notes.txt"
    assert safe_filename("") == "file"
