"""Recovering a PDF from a PKCS#7 signature container."""

from docflow.adapters.pdf.unwrap import unwrap_pkcs7_pdf


def _pdf(body: bytes = b"body") -> bytes:
    return b"%PDF-1.7\n" + body + b"\n%%EOF"


def test_plain_pdf_is_left_alone():
    assert unwrap_pkcs7_pdf(_pdf()) is None


def test_not_a_pdf_at_all_is_left_alone():
    # Deliberately not an error: the caller passes the original bytes on and
    # lets the server report what is actually wrong with the file.
    assert unwrap_pkcs7_pdf(b"\x30\x82\x0a\x0b just some ASN.1 and nothing else") is None


def test_pdf_embedded_after_an_asn1_header_is_extracted():
    # The shape of the real files: a DER header, the PDF, then the signature.
    prefix = b"\x30\x82" + b"\x00" * 69
    signature_tail = b"\x30\x82 certificate bytes"
    extracted = unwrap_pkcs7_pdf(prefix + _pdf() + signature_tail)

    assert extracted == _pdf()
    assert extracted is not None and extracted.startswith(b"%PDF-")
    assert extracted.endswith(b"%%EOF")


def test_trailing_bytes_after_the_last_eof_are_dropped():
    extracted = unwrap_pkcs7_pdf(b"wrapper" + _pdf() + b"trailing signature junk")
    assert extracted == _pdf()


def test_the_last_eof_wins_when_the_pdf_has_several():
    inner = b"%PDF-1.7\nfirst revision\n%%EOF\nincremental update\n%%EOF"
    assert unwrap_pkcs7_pdf(b"wrapper" + inner) == inner


def test_missing_eof_keeps_everything_from_the_header_on():
    truncated = b"wrapper" + b"%PDF-1.7\nno end marker here"
    assert unwrap_pkcs7_pdf(truncated) == b"%PDF-1.7\nno end marker here"


def test_an_eof_before_the_pdf_header_is_not_mistaken_for_the_end():
    # A stray %%EOF in the wrapper must not truncate the document to nothing.
    data = b"junk %%EOF junk " + b"%PDF-1.7\nreal content"
    assert unwrap_pkcs7_pdf(data) == b"%PDF-1.7\nreal content"
