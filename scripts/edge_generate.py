import cv2, os

def generate_edges(in_path, out_path):
    cap = cv2.VideoCapture(in_path)
    assert cap.isOpened(), "Could not open input video."
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bright = 50
    contrast = 1.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # use "avc1" if you prefer H.264 and it's available
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=bright)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.4)
        edges = cv2.Canny(blurred, 10, 50)
        out.write(edges)
    cap.release()
    out.release()


def generate_edges_from_images(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    bright = 50
    contrast = 1.0

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(f for f in os.listdir(in_dir) if os.path.splitext(f)[1].lower() in exts)
    assert files, f"No images found in {in_dir}"

    for fname in files:
        frame = cv2.imread(os.path.join(in_dir, fname))
        assert frame is not None, f"Could not read {fname}"

        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=bright)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.4)
        edges = cv2.Canny(blurred, 10, 50)
        cv2.imwrite(os.path.join(out_dir, fname), edges)

    print(f"Done: {len(files)} images → {out_dir}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_dir = os.path.join(script_dir, "..", "dataset", "CAM_F0")
    out_dir = os.path.join(script_dir, "..", "dataset", "CAM_F0_edge")
    generate_edges_from_images(in_dir, out_dir)