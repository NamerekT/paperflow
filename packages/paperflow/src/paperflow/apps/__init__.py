"""The executables: the webhook service, the folder watcher, the CLI tools.

Everything below this package is composition and process lifecycle - reading
settings, wiring adapters to the engine, starting threads, exposing an ASGI app or a
console entry point. None of it is imported by anything outside `apps/` itself.
"""
