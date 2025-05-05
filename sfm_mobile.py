import os
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

class IMUAssistedSfM:
    def __init__(self, image_dir, imu_json, output_dir,
                 min_matches=20, ransac_thresh=4.0):
        self.image_dir, self.imu_json, self.output_dir = image_dir, imu_json, output_dir
        os.makedirs(output_dir, exist_ok=True)

        # data containers
        self.images, self.names = [], []
        self.imu, self.K = {}, {}
        self.features, self.matches = {}, {}
        self.poses, self.cloud = {}, []

        # params
        self.min_matches, self.ransac_thresh = min_matches, ransac_thresh
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()

    # ---------- loading ----------
    def load_data(self):
        # images
        files = sorted(f for f in os.listdir(self.image_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        for f in files:
            img = cv2.imread(os.path.join(self.image_dir, f))
            if img is None: continue
            self.images.append(img); self.names.append(f)

        # IMU JSON (optional per image)
        with open(self.imu_json) as f:
            self.imu = json.load(f)

        # intrinsics ≈ focal = 1.2·max(h,w)
        for i, img in enumerate(self.images):
            h, w = img.shape[:2]; f = 1.2 * max(h, w)
            self.K[i] = np.array([[f, 0, w/2],
                                  [0, f, h/2],
                                  [0, 0,   1]])

    # ---------- feature extraction ----------
    def extract_features(self):
        for i, img in enumerate(self.images):
            kp, des = self.detector.detectAndCompute(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            self.features[i] = dict(kp=kp, des=des)

    # ---------- IMU helper ----------
    def _imu_relative(self, i, j):
        ai = self.imu.get(self.names[i], {})
        aj = self.imu.get(self.names[j], {})
        # default: identity pose + zero translation
        for a in (ai, aj):
            a.setdefault('rotation', [0, 0, 0])
            a.setdefault('position', [0, 0, 0])
            a.setdefault('timestamp', 0)

        Ri = Rotation.from_euler('xyz', ai['rotation'], True).as_matrix()
        Rj = Rotation.from_euler('xyz', aj['rotation'], True).as_matrix()
        Rrel = Rj @ Ri.T
        trel = (Ri.T @ (np.array(aj['position']) - np.array(ai['position']))).reshape(3, 1)
        conf = 1.0 / (1.0 + abs(aj['timestamp'] - ai['timestamp']))
        return Rrel, trel, conf

    # ---------- pairwise matching ----------
    def match_pairs(self):
        n = len(self.images)
        for i in range(n):
            for j in range(i+1, n):
                Rrel, trel, _ = self._imu_relative(i, j)

                m = self.matcher.knnMatch(self.features[i]['des'],
                                          self.features[j]['des'], k=2)
                good = [x[0] for x in m if x[0].distance < 0.75 * x[1].distance]
                if len(good) < self.min_matches: continue

                pts1 = np.float32([self.features[i]['kp'][g.queryIdx].pt for g in good])
                pts2 = np.float32([self.features[j]['kp'][g.trainIdx].pt for g in good])

                E, mask = cv2.findEssentialMat(pts1, pts2, self.K[i],
                                               cv2.RANSAC, 0.999, self.ransac_thresh)
                if E is None: continue
                mask = mask.ravel().astype(bool)
                inl = [good[k] for k in range(len(good)) if mask[k]]
                if len(inl) < self.min_matches: continue

                self.matches[(i, j)] = dict(matches=inl,
                                            pts1=pts1[mask], pts2=pts2[mask],
                                            E=E, Rrel=Rrel, trel=trel)

    # ---------- initial pair ----------
    def init_poses(self):
        if not self.matches:
            raise RuntimeError("no image pairs with enough inliers")
        i, j = max(self.matches, key=lambda k: len(self.matches[k]['matches']))
        rec = self.matches[(i, j)]

        self.poses[i] = dict(R=np.eye(3), t=np.zeros((3, 1)))
        _, R, t, _ = cv2.recoverPose(rec['E'], rec['pts1'], rec['pts2'], self.K[i])
        scale = np.linalg.norm(rec['trel']) / (np.linalg.norm(t) + 1e-9)
        self.poses[j] = dict(R=R, t=t * scale)
        self._triangulate(i, j)

    # ---------- triangulation ----------
    def _triangulate(self, i, j):
        Mi = self.K[i] @ np.hstack((self.poses[i]['R'], self.poses[i]['t']))
        Mj = self.K[j] @ np.hstack((self.poses[j]['R'], self.poses[j]['t']))
        rec = self.matches[(i, j)]
        P4 = cv2.triangulatePoints(Mi, Mj, rec['pts1'].T, rec['pts2'].T)
        P3 = (P4[:3] / P4[3]).T
        for k, X in enumerate(P3):
            m = rec['matches'][k]
            self.cloud.append(dict(X=X, track=[(i, m.queryIdx), (j, m.trainIdx)]))

    # ---------- incremental PnP ----------
    def add_cameras(self):
        while True:
            remaining = set(range(len(self.images))) - set(self.poses)
            best, best_cnt = None, 0
            for i in remaining:
                cnt = sum(any(img == i for img, _ in pt['track']) for pt in self.cloud)
                if cnt > best_cnt:
                    best, best_cnt = i, cnt
            if best is None or best_cnt < self.min_matches:
                break
            self._solve_pnp(best)

    def _solve_pnp(self, i):
        obj, imgp = [], []
        for pt in self.cloud:
            for img_idx, feat_idx in pt['track']:
                if img_idx == i:
                    obj.append(pt['X'])
                    imgp.append(self.features[i]['kp'][feat_idx].pt)
        if len(obj) < self.min_matches:
            return
        obj, imgp = np.asarray(obj), np.asarray(imgp)
        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            obj, imgp, self.K[i], None, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            self.poses[i] = dict(R=R, t=tvec.reshape(3, 1))

    # ---------- COLMAP export ----------
    def save_colmap(self):
        # cameras
        with open(os.path.join(self.output_dir, 'cameras.txt'), 'w', newline='\n') as f:
            f.write("# Cameras\n# Number of cameras: {}\n".format(len(self.K)))
            for idx, K in self.K.items():
                h, w = self.images[idx].shape[:2]
                fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
                f.write(f"\n{idx} SIMPLE_PINHOLE {w} {h} {fx} {cx} {cy}")

        # images
        with open(os.path.join(self.output_dir, 'images.txt'), 'w', newline='\n') as f:
            f.write("# Images\n# Number of images: {}\n".format(len(self.poses)))
            for idx, pose in self.poses.items():
                R, t = pose['R'], pose['t'].ravel()
                q = Rotation.from_matrix(R).as_quat()  # x,y,z,w
                f.write(f"\n{idx} {q[3]} {q[0]} {q[1]} {q[2]} "
                        f"{t[0]} {t[1]} {t[2]} {idx} {self.names[idx]}\n")

        # points
        with open(os.path.join(self.output_dir, 'points3D.txt'), 'w', newline='\n') as f:
            f.write("# Points3D\n# Number of points: {}\n".format(len(self.cloud)))
            for pid, pt in enumerate(self.cloud):
                X = pt['X']; track = " ".join(f"{i} {fi}" for i, fi in pt['track'])
                f.write(f"\n{pid} {X[0]} {X[1]} {X[2]} 128 128 128 0 {track}")

    # ---------- pipeline ----------
    def run(self):
        self.load_data()
        self.extract_features()
        self.match_pairs()
        self.init_poses()
        self.add_cameras()
        self.save_colmap()
        print("SfM finished – results in", self.output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IMU-Assisted SfM Pipeline")
    parser.add_argument('--image_dir', required=True, help="Directory of input images")
    parser.add_argument('--imu_json', required=True, help="Path to IMU JSON file")
    parser.add_argument('--output_dir', required=True, help="Directory to save outputs")
    args = parser.parse_args()

    sfm = IMUAssistedSfM(
        image_dir=args.image_dir,
        imu_json=args.imu_json,
        output_dir=args.output_dir
    )
    sfm.run()
