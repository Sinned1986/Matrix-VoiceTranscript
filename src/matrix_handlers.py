from __future__ import annotations

import asyncio

import contextlib
import logging
from collections import defaultdict

from nio import (
    AsyncClient,
    InviteMemberEvent,
    JoinError,
    JoinResponse,
    KeysQueryResponse,
    KeyVerificationCancel,
    KeyVerificationEvent,
    KeyVerificationKey,
    KeyVerificationMac,
    KeyVerificationStart,
    LocalProtocolError,
    MegolmEvent,
    RoomMessageAudio,
    RoomMessageFile,
    RoomEncryptedAudio,
    RoomEncryptedFile,
    SyncResponse,
    ToDeviceError,
    ToDeviceMessage,
    UnknownEncryptedEvent,
    UnknownToDeviceEvent
)

from nio.exceptions import EncryptionError, LocalProtocolError

import os

from src.bot_service import BotService

logger = logging.getLogger(__name__)


def _own_mxid(client: AsyncClient) -> str:
    uid = (getattr(client, "user_id", "") or "").strip()
    if uid:
        return uid
    return (getattr(client, "user", "") or "").strip()


verification_devices = {}
verification_done_sent = set()

def register_matrix_callbacks(client: AsyncClient, service: BotService) -> None:
    first_sync_done = False

    def _trust_all_devices() -> None:
        # Auto-trust every device we learn about. The bot operates in untrusted
        # public rooms; refusing keys to unverified devices would block all decryption.
        own_user = client.user_id
        own_device = client.device_id
        for user_id, devices in client.device_store.items():
            for device_id, olm_device in devices.items():
                if user_id == own_user and device_id == own_device:
                    continue
                if olm_device.verified or olm_device.ignored:
                    continue
                client.verify_device(olm_device)

    async def _claim_missing_olm_sessions() -> None:
        # nio's sync_forever only auto-claims for "wedged" devices; a receive-only
        # bot must explicitly claim one-time keys to establish Olm sessions,
        # otherwise senders can't share Megolm room keys with us.
        to_claim: dict[str, list[str]] = defaultdict(list)
        for user_id, devices in client.device_store.items():
            if user_id == client.user_id:
                continue
            for device_id, olm_device in devices.items():
                if client.olm.session_store.get(olm_device.curve25519):
                    continue
                to_claim[user_id].append(device_id)
        if not to_claim:
            return
        logger.info("Claiming one-time keys for %d users", len(to_claim))
        try:
            await client.keys_claim(dict(to_claim))
            await client.send_to_device_messages()
        except Exception:
            logger.exception("keys_claim failed")

    async def on_first_sync(_response: SyncResponse) -> None:
        nonlocal first_sync_done
        if first_sync_done:
            return
        first_sync_done = True
        me = _own_mxid(client)
        for room_id in list(getattr(client, "invited_rooms", {}).keys()):
            if room_id in client.rooms:
                continue
            try:
                logger.info("Auto-joining residual invite: %s (mxid=%s)", room_id, me)
                resp = await client.join(room_id)
                if isinstance(resp, JoinError):
                    logger.error("Join failed: %s %s", room_id, resp.message)
            except Exception:
                logger.exception("Auto-join failed for %s", room_id)
        _trust_all_devices()
        await _claim_missing_olm_sessions()

    async def on_keys_query(_response: KeysQueryResponse) -> None:
        # Newly-discovered devices need to be trusted so we can establish Olm sessions
        _trust_all_devices()
        await _claim_missing_olm_sessions()

    async def on_encrypted(room: object, event: MegolmEvent) -> None:
        try:
            decrypted = await client.decrypt_event(event)
        except EncryptionError:
            logger.debug("Cannot decrypt event yet, requesting keys room=%s", room.room_id)  # type: ignore[attr-defined]
            # First make sure we have Olm sessions with the sender's devices,
            # then ask for the room key.
            await _claim_missing_olm_sessions()
            with contextlib.suppress(LocalProtocolError):
                await client.request_room_key(event)
            return
        if isinstance(decrypted, (RoomMessageAudio, RoomMessageFile)):
            try:
                await service.handle_audio_event(room, decrypted)
            except Exception:
                logger.exception("Error handling encrypted audio room=%s", room.room_id)  # type: ignore[attr-defined]

    async def on_invite(room: object, event: InviteMemberEvent) -> None:
        me = _own_mxid(client)
        if event.state_key != me:
            return
        room_id = room.room_id  # type: ignore[attr-defined]
        logger.info("Accepting invite to %s", room_id)
        resp = await client.join(room_id)
        if isinstance(resp, JoinError):
            logger.error("Join failed: %s %s", room_id, resp.message)
        elif isinstance(resp, JoinResponse):
            logger.info("Joined %s", room_id)

    async def on_audio(room: object, event: RoomMessageAudio) -> None:
        try:
            await service.handle_audio_event(room, event)
        except Exception:
            logger.exception("Error handling audio event room=%s", room.room_id)  # type: ignore[attr-defined]

    async def on_file(room: object, event: RoomMessageFile) -> None:
        info: dict = event.source.get("content", {}).get("info") or {}
        mime: str = info.get("mimetype", "")
        if not mime.startswith("audio/"):
            return
        try:
            await service.handle_audio_event(room, event)
        except Exception:
            logger.exception("Error handling file event room=%s", room.room_id)  # type: ignore[attr-defined]

    async def on_key_verification(event):
        # For this bot, only allow verification with another session
        # belonging to the transcriptbot account itself.
        if event.sender != client.user_id:
            logger.warning(
                "Ignoring SAS verification from another user: %s",
                event.sender,
            )
            return

        if isinstance(event, KeyVerificationStart):
            logger.info(
                "SAS verification started by %s, device=%s, tx=%s",
                event.sender,
                event.from_device,
                event.transaction_id,
            )

            if "emoji" not in event.short_authentication_string:
                logger.error("Other device does not support emoji SAS")
                await client.cancel_key_verification(
                    event.transaction_id,
                    reject=False,
                )
                return

            response = await client.accept_key_verification(
                event.transaction_id
            )

            if isinstance(response, ToDeviceError):
                logger.error(
                    "accept_key_verification failed: %s",
                    response,
                )
                return

            sas = client.key_verifications[event.transaction_id]

            response = await client.to_device(sas.share_key())

            if isinstance(response, ToDeviceError):
                logger.error(
                    "Sending SAS key failed: %s",
                    response,
                )
            verification_devices[event.transaction_id] = event.from_device

        elif isinstance(event, KeyVerificationKey):
            sas = client.key_verifications[event.transaction_id]

            emoji = sas.get_emoji()

            logger.warning("==========================================")
            logger.warning("MATRIX SAS VERIFICATION")
            logger.warning("Transaction: %s", event.transaction_id)

            for symbol, name in emoji:
                logger.warning("%s  %s", symbol, name)

            logger.warning("")
            logger.warning(
                "If these emojis match Element, run:"
            )
            logger.warning(
                "docker compose exec matrix-voice-transcript "
                "touch /tmp/sas-confirm"
            )
            logger.warning("==========================================")

            # Wait up to 2 minutes for manual confirmation.
            for _ in range(120):
                if os.path.exists("/tmp/sas-confirm"):
                    os.unlink("/tmp/sas-confirm")

                    logger.info("SAS emojis manually confirmed")

                    response = await client.confirm_short_auth_string(
                                       event.transaction_id
                    )

                    if isinstance(response, ToDeviceError):
                        logger.error(
                            "confirm_short_auth_string failed: %s",
                            response,
                        )

                    return

                await asyncio.sleep(1)

            logger.warning("SAS verification timed out")

            await client.cancel_key_verification(
                event.transaction_id,
                reject=False,
            )

        elif isinstance(event, KeyVerificationMac):
            sas = client.key_verifications.get(event.transaction_id)

            if sas is None:
                logger.warning(
                    "Received SAS MAC for unknown transaction %s",
                    event.transaction_id,
                )
                return

            try:
                message = sas.get_mac()
            except LocalProtocolError as exc:
                logger.error("SAS protocol error: %s", exc)
                return

            response = await client.to_device(message)

            if isinstance(response, ToDeviceError):
                logger.error("Sending SAS MAC failed: %s", response)
                return

            logger.info(
                "SAS verification finished: verified=%s devices=%s",
                sas.verified,
                sas.verified_devices,
            )

            if sas.verified:
                other_device = verification_devices.get(event.transaction_id)

                if not other_device:
                    logger.error(
                        "Cannot send verification done: no device known for tx=%s",
                        event.transaction_id,
                    )
                    return

                done = ToDeviceMessage(
                    "m.key.verification.done",
                    event.sender,
                    other_device,
                    {
                        "transaction_id": event.transaction_id,
                    },
                )

            response = await client.to_device(done)

            if isinstance(response, ToDeviceError):
                logger.error(
                    "Sending m.key.verification.done failed: %s",
                    response,
                )
                return

            verification_done_sent.add(event.transaction_id)

            logger.info(
                "Sent m.key.verification.done tx=%s device=%s",
                event.transaction_id,
                other_device,
            )
        elif isinstance(event, KeyVerificationCancel):
            logger.warning(
                "SAS verification cancelled by %s: %s",
                event.sender,
                event.reason,
            )

    async def on_unknown_to_device(event):
        if not isinstance(event, UnknownToDeviceEvent):
            return

        logger.info(
            "Received to-device event type=%s sender=%s",
            event.type,
            event.sender,
        )

        if event.type == "m.key.verification.done":
            content = event.source.get("content", {})
            transaction_id = content.get("transaction_id")

            logger.info(
                "Received m.key.verification.done tx=%s",
                transaction_id,
            )

            verification_devices.pop(transaction_id, None)
            verification_done_sent.discard(transaction_id)
            return

        if event.type != "m.key.verification.request":
            return

        # Only allow another device belonging to this same bot account.
        if event.sender != client.user_id:
            logger.warning(
                "Ignoring verification request from different user: %s",
                event.sender,
            )
            return

        content = event.source.get("content", {})

        transaction_id = content.get("transaction_id")
        from_device = content.get("from_device")
        methods = content.get("methods", [])

        verification_devices[transaction_id] = from_device

        if not transaction_id or not from_device:
            logger.warning(
                "Malformed verification request: %r",
                content,
            )
            return

        if "m.sas.v1" not in methods:
            logger.warning(
                "Verification request does not support SAS: %r",
                methods,
            )
            return

        logger.warning(
            "Accepting SAS verification request from device=%s tx=%s",
            from_device,
            transaction_id,
        )

        ready = ToDeviceMessage(
            "m.key.verification.ready",
            event.sender,
            from_device,
            {
                "from_device": client.device_id,
                "methods": ["m.sas.v1"],
                "transaction_id": transaction_id,
            },
        )

        response = await client.to_device(ready)

        logger.info(
            "Sent m.key.verification.ready: %s",
            response,
        )

    client.add_response_callback(on_first_sync, SyncResponse)
    client.add_response_callback(on_keys_query, KeysQueryResponse)
    client.add_event_callback(on_invite, InviteMemberEvent)
    client.add_event_callback(on_audio, RoomMessageAudio)
    client.add_event_callback(on_audio, RoomEncryptedAudio)
    client.add_event_callback(on_file, RoomMessageFile)
    client.add_event_callback(on_file, RoomEncryptedFile)
    client.add_event_callback(on_encrypted, MegolmEvent)
    client.add_event_callback(on_encrypted, UnknownEncryptedEvent)

    client.add_to_device_callback(
        on_unknown_to_device,
        (UnknownToDeviceEvent,),
    )
    client.add_to_device_callback(
        on_key_verification,
        (KeyVerificationEvent,),
    )
