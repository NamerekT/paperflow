"""Recovering a PDF from a PKCS#7 signature container.

Pure byte handling, no I/O: it takes bytes and returns bytes, which is what
makes it the one piece of this system worth a test with no infrastructure
behind it.

The case, found on three bank statements at once: Paperless rejected them with
"Unsupported mime type application/octet-stream". The files do not begin with
"%PDF-". Their first bytes are ASN.1 DER carrying OID 1.2.840.113549.1.7.2 -
PKCS#7 SignedData - because they are PDFs wrapped in a qualified electronic
signature container and saved with a .pdf extension without unwrapping. The
PDF inside is intact and reads perfectly; "%PDF-" started at byte 71 in all
three, with a proper "%%EOF" later in the file, followed by the certificate
and signature bytes.

Both python-magic, which Paperless uses, and ordinary PDF readers look at the
first bytes of the file. They see ASN.1, not "%PDF-", and honestly report
application/octet-stream even though the content is valid.
"""

PDF_MAGIC = b"%PDF-"
PDF_EOF_MARKER = b"%%EOF"


def unwrap_pkcs7_pdf(data: bytes) -> bytes | None:
    """Return the PDF embedded in `data`, or None if it does not need extracting.

    None means "use the original bytes": either the file already starts with
    "%PDF-", or it contains no PDF header at all. The second case is not
    treated as an error here - the file is genuinely not a PDF, and it is
    better to let it fail downstream with the server's own clear message than
    to invent one.
    """
    if data.startswith(PDF_MAGIC):
        return None

    pdf_start = data.find(PDF_MAGIC)
    if pdf_start == -1:
        return None

    # Trailing bytes after %%EOF are the signature payload. Cut there when the
    # marker exists and follows the header; a file whose only %%EOF precedes
    # the PDF header has none of its own, so keep everything to the end.
    pdf_end = data.rfind(PDF_EOF_MARKER)
    if pdf_end == -1 or pdf_end < pdf_start:
        return data[pdf_start:]
    return data[pdf_start : pdf_end + len(PDF_EOF_MARKER)]
