
import asyncio
import json

import numpy as np
import sounddevice as sd
import websockets


# ============================================================
# CONFIGURATION
# ============================================================

SERVER_URL = "ws://127.0.0.1:5000/live"

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS


def record_chunk():
    """Record two seconds of mono int16 PCM from the microphone."""

    print("Listening...", end="\r", flush=True)

    audio = sd.rec(
        frames=CHUNK_SAMPLES,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocking=True,
    )

    # Explicit little-endian int16 for the server protocol.
    return np.asarray(audio).reshape(-1).astype("<i2")


async def send_audio(ws):
    """Continuously send new microphone audio."""

    try:
        while True:
            # Keep blocking microphone capture off the event loop.
            chunk = await asyncio.to_thread(record_chunk)

            # Send binary PCM, NOT a JSON list of samples.
            await ws.send(chunk.tobytes())

    except asyncio.CancelledError:
        raise


async def receive_predictions(ws):
    """Receive and display classification results."""

    async for message in ws:
        result = json.loads(message)

        if "error" in result:
            print("\nServer error:", result)
            continue

        print(
            f"\nAudio through: {result['audio_end_seconds']:.2f}s"
            f"\nPrediction:    {result['label'].upper()}"
            f"\nProbability:   {result['probability']:.4f}"
            f"\nWindow:        {result['window_seconds']} seconds"
            f"\nHop:           {result['hop_seconds']} seconds"
        )

        if result["label"] == "distress":
            print("⚠️ Possible distress detected.")

        print("-" * 40)


async def main():
    print("Connecting to WASP...")
    print("Server:", SERVER_URL)

    try:
        async with websockets.connect(
            SERVER_URL,
            max_size=1024 * 1024,
        ) as ws:

            print("Connected. Microphone streaming started.")
            print("Press Ctrl+C to stop.\n")

            sender = asyncio.create_task(send_audio(ws))
            receiver = asyncio.create_task(
                receive_predictions(ws)
            )

            try:
                done, pending = await asyncio.wait(
                    {sender, receiver},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    # Propagate a genuine client/server error.
                    task.result()

            finally:
                for task in (sender, receiver):
                    if not task.done():
                        task.cancel()

                await asyncio.gather(
                    sender, receiver,
                    return_exceptions=True,
                )

    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\nConnection or audio error: {exc}")

    print("\nWASP microphone client stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
