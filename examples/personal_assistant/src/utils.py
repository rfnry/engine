from rfnry_engine import Document, Image


def attachment(media_type: str, name: str, data: bytes) -> Image | Document:
    if media_type.startswith("image/"):
        return Image(media_type=media_type, data=data)
    return Document(media_type=media_type, name=name, data=data)
