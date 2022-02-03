from pathlib import Path

from automl_common.backend.stores.store import Store


class MockDirStore(Store[str]):
    """A Mock store that uses directories"""

    def path(self, key: str) -> Path:
        """Path to the int"""
        return self.dir / str(key)

    def load(self, key: str) -> str:
        """Load a str"""
        item_path = self.path(key) / "item"
        return item_path.read_text()

    def save(self, item: str, key: str) -> None:
        """Save a str"""
        dir = self.dir / str(key)
        if not dir.exists():
            dir.mkdir()

        item_path = self.path(key) / "item"
        item_path.write_text(item)
