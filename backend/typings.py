from typing import Any, Dict, List, Optional, Union

# Type stubs for pytesseract
class PyTesseract:
    def image_to_string(self, image: Any, lang: Optional[str] = None, **kwargs) -> str:
        ...

# Type stubs for fitz (PyMuPDF)
class Page:
    def get_text(self) -> str:
        ...
    def get_pixmap(self) -> Any:
        ...

class Document:
    def load_page(self, page_num: int) -> Page:
        ...
    def close(self) -> None:
        ...

class Fitz:
    def open(self, path: str) -> Document:
        ...

# Type stubs for pytube
class Caption:
    def generate_srt_captions(self) -> str:
        ...

class Captions:
    def get_by_language_code(self, code: str) -> Optional[Caption]:
        ...

class YouTube:
    title: str
    description: str
    captions: Captions
    def __init__(self, url: str) -> None:
        ...

# Type stubs for ikapi
class FileStorage:
    def __init__(self, data_dir: str) -> None:
        ...

class IKApi:
    def __init__(self, args: Any, storage: FileStorage) -> None:
        ...
    def search(self, q: str, pagenum: int, maxpages: int) -> str:
        ...
    def fetch_doc(self, docid: str) -> str:
        ...
