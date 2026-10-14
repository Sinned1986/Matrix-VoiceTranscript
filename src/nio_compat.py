
from nio.events.room_events import RoomMessage


def patch_forwarded_media():
    """Allow unencrypted audio attachments in E2EE room events."""

    if getattr(RoomMessage, "_forwarded_media_patched", False):
        return

    original_parser = RoomMessage.parse_decrypted_event

    @classmethod
    def parse_decrypted(cls, event_dict):
        content = event_dict.get("content") or {}

        if (
            content.get("msgtype") in {"m.audio", "m.file"}
            and "file" not in content
            and isinstance(content.get("url"), str)
        ):
            # The room event was encrypted, but the media is not.
            # Parse it as ordinary media with its normal schema.
            return cls.parse_event(event_dict)

        return original_parser(event_dict)

    RoomMessage.parse_decrypted_event = parse_decrypted
    RoomMessage._forwarded_media_patched = True
