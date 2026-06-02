import socket, json, traceback, threading, os

SOCK_PATH = "/tmp/cc_ipc.sock"
SEP = b"\n##END##\n"


def _exec(code):
    ns = {}
    try:
        import pycc
        ns["pycc"] = pycc
        ns["CC"] = pycc.GetInstance()
        try:
            import cccorelib
            ns["cccorelib"] = cccorelib
        except ImportError:
            pass
        exec(compile(code, "<cc_ipc>", "exec"), ns)
        return True, str(ns.get("_result", None))
    except Exception:
        return False, traceback.format_exc()


def _handle(conn):
    try:
        buf = b""
        while SEP not in buf:
            chunk = conn.recv(8192)
            if not chunk:
                break
            buf += chunk
        msg = json.loads(buf.split(SEP)[0])
        ok, result = _exec(msg["code"])
        conn.sendall(json.dumps({"ok": ok, "result": result}).encode() + SEP)
    except Exception as exc:
        try:
            conn.sendall(json.dumps({"ok": False, "result": str(exc)}).encode() + SEP)
        except Exception:
            pass
    finally:
        conn.close()


def _serve(srv):
    print("[cc_ipc] ready")
    while True:
        try:
            conn, _ = srv.accept()
            threading.Thread(target=_handle, args=(conn,), daemon=True).start()
        except Exception as exc:
            print("[cc_ipc] stopped:", exc)
            break


if os.path.exists(SOCK_PATH):
    os.unlink(SOCK_PATH)

_srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
_srv.bind(SOCK_PATH)
_srv.listen(8)
threading.Thread(target=_serve, args=(_srv,), daemon=True).start()
print("[cc_ipc] listening on", SOCK_PATH)
