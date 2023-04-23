
import struct

PAYLOAD_HEADER_SIZE_BYTES = 2
BUFFER_CHUNK_SIZE_BYTES = 8192

def socket_write(socket, message_string):
    message_length_header = struct.pack('<H', len(message_string))

    actual_message = message_string.encode('ascii')
    final_message_bytes = bytearray(PAYLOAD_HEADER_SIZE_BYTES + len(actual_message))

    final_message_bytes[:PAYLOAD_HEADER_SIZE_BYTES] = message_length_header
    final_message_bytes[PAYLOAD_HEADER_SIZE_BYTES:] = actual_message

    progress = 0

    while progress < len(final_message_bytes):
        amount_to_send = BUFFER_CHUNK_SIZE_BYTES
        # If it would overflow, instead read the remainder
        if len(final_message_bytes) - progress < BUFFER_CHUNK_SIZE_BYTES:
            amount_to_send = len(final_message_bytes) - progress

        socket.send(final_message_bytes[progress:progress + amount_to_send])
        progress += amount_to_send

def socket_read(socket) -> str:
    # First, read the message length header (2 bytes)
    message_length_header_bytes = socket.recv(PAYLOAD_HEADER_SIZE_BYTES)
    message_length_header = struct.unpack('<H', message_length_header_bytes)[0]

    # Next, read the actual message
    actual_message_bytes = b''
    remaining_bytes_to_read = message_length_header
    while remaining_bytes_to_read > 0:
        chunk = socket.recv(min(BUFFER_CHUNK_SIZE_BYTES, remaining_bytes_to_read))
        actual_message_bytes += chunk
        remaining_bytes_to_read -= len(chunk)

    actual_message = actual_message_bytes.decode('ascii')
    return actual_message