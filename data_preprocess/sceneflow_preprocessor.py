import os
from .data_preprocessor import DataPreprocessor


class SceneFlowPreprocessor(DataPreprocessor):
    """
    Preprocessor for the Scene Flow dataset
    """

    def __init__(self, dataset_path, max_disp, block_sz, match_method, device, full_ZSAD):
        """
        Preprocessor for the Scene Flow dataset

        :param dataset_path: directory to the dataset
        :param max_disp: maximum disparity to check when matching
        :param block_sz: block size for the stereo algorithm
        :param match_method: choose between local_BM and SG_BM for local or semi-global block matching
        :param device: device to compute confidence measures, choose between cuda and cpu
        :param full_ZSAD: if set to True, ZSAD is performed on a 3x3 window. Otherwise, it is performed on a partial 3x3 window
        """
        super(SceneFlowPreprocessor, self).__init__(dataset_path, max_disp, block_sz, match_method, device, full_ZSAD)
        self.l_im_path = None
        self.r_im_path = None
        self.disp_path = None
        self.conf_path = None
        self.train_list = []
        self.test_list = []

    def _preprocess_flyingthings(self):
        """
        Preprocess the FlyingThings3D subset in Scene Flow

        :return: None
        """
        subset = "flyingthings3d"
        subset_path = os.path.join(self.dataset_path, subset)
        splits = os.listdir(os.path.join(subset_path, "frames_finalpass"))
        for split in splits:
            scenes = os.listdir(os.path.join(subset_path, "frames_finalpass", split))
            for scene in scenes:
                sub_scenes = os.listdir(os.path.join(subset_path, "frames_finalpass", split, scene))
                for sub_scene in sub_scenes:
                    scene_path_to_save = subset + " " + os.path.join(split, scene, sub_scene) + " "
                    self.l_im_path = os.path.join(subset_path, "frames_finalpass", split, scene, sub_scene, "left")
                    self.r_im_path = os.path.join(subset_path, "frames_finalpass", split, scene, sub_scene, "right")
                    self.disp_path = os.path.join(subset_path, "raw_disp", split, scene, sub_scene)
                    self.conf_path = os.path.join(subset_path, "conf", split, scene, sub_scene)
                    if not os.path.exists(self.disp_path):
                        os.makedirs(self.disp_path)
                    if not os.path.exists(self.conf_path):
                        os.makedirs(self.conf_path)

                    frames = os.listdir(self.l_im_path)
                    frame_count = 0
                    for f in frames:
                        frame_count += 1
                        print("Processing frame %d/%d in %s subset %s split %s scene %s sub-scene" % (
                            frame_count, len(frames), subset, split, scene, sub_scene))
                        l_im_frame = os.path.join(self.l_im_path, f)
                        r_im_frame = os.path.join(self.r_im_path, f)
                        disp_frame = os.path.join(self.disp_path, f.replace(".png", ".npy"))
                        conf_frame = os.path.join(self.conf_path, f.replace(".png", ".npy"))
                        self._process_frame(l_im_frame, r_im_frame, disp_frame, conf_frame)
                        frame_to_save = scene_path_to_save + f
                        if split == "TRAIN":
                            self.train_list.append(frame_to_save)
                        else:
                            self.test_list.append(frame_to_save)

    def _preprocess_monkaa(self):
        """
        Preprocess the Monkaa subset in Scene Flow

        :return: None
        """
        subset = "monkaa"
        subset_path = os.path.join(self.dataset_path, subset)
        scenes = os.listdir(os.path.join(subset_path, "frames_finalpass"))
        for scene in scenes:
            scene_path_to_save = subset + " " + scene + " "
            self.l_im_path = os.path.join(subset_path, "frames_finalpass", scene, "left")
            self.r_im_path = os.path.join(subset_path, "frames_finalpass", scene, "right")
            self.disp_path = os.path.join(subset_path, "raw_disp", scene)
            self.conf_path = os.path.join(subset_path, "conf", scene)
            if not os.path.exists(self.disp_path):
                os.makedirs(self.disp_path)
            if not os.path.exists(self.conf_path):
                os.makedirs(self.conf_path)

            frames = os.listdir(self.l_im_path)
            frame_count = 0
            for f in frames:
                frame_count += 1
                print("Processing frame %d/%d in %s subset %s scene" % (frame_count, len(frames), subset, scene))
                l_im_frame = os.path.join(self.l_im_path, f)
                r_im_frame = os.path.join(self.r_im_path, f)
                disp_frame = os.path.join(self.disp_path, f.replace(".png", ".npy"))
                conf_frame = os.path.join(self.conf_path, f.replace(".png", ".npy"))
                self._process_frame(l_im_frame, r_im_frame, disp_frame, conf_frame)
                frame_to_save = scene_path_to_save + f
                self.train_list.append(frame_to_save)

    def _preprocess_driving(self):
        """
        Preprocess the Driving subset in Scene Flow

        :return: None
        """
        subset = "driving"
        subset_path = os.path.join(self.dataset_path, subset)
        focals = os.listdir(os.path.join(subset_path, "frames_finalpass"))
        for focal in focals:
            directions = os.listdir(os.path.join(subset_path, "frames_finalpass", focal))
            for direction in directions:
                speeds = os.listdir(os.path.join(subset_path, "frames_finalpass", focal, direction))
                for speed in speeds:
                    scene_path_to_save = subset + " " + os.path.join(focal, direction, speed) + " "
                    self.l_im_path = os.path.join(subset_path, "frames_finalpass", focal, direction, speed, "left")
                    self.r_im_path = os.path.join(subset_path, "frames_finalpass", focal, direction, speed, "right")
                    self.disp_path = os.path.join(subset_path, "raw_disp", focal, direction, speed)
                    self.conf_path = os.path.join(subset_path, "conf", focal, direction, speed)
                    if not os.path.exists(self.disp_path):
                        os.makedirs(self.disp_path)
                    if not os.path.exists(self.conf_path):
                        os.makedirs(self.conf_path)

                    frames = os.listdir(self.l_im_path)
                    frame_count = 0
                    for f in frames:
                        frame_count += 1
                        print("Preprocessing frame %d/%d in %s subset %s focal length %s direction %s speed" % (
                            frame_count, len(frames), subset, focal, direction, speed))
                        l_im_frame = os.path.join(self.l_im_path, f)
                        r_im_frame = os.path.join(self.r_im_path, f)
                        disp_frame = os.path.join(self.disp_path, f.replace(".png", ".npy"))
                        conf_frame = os.path.join(self.conf_path, f.replace(".png", ".npy"))
                        self._process_frame(l_im_frame, r_im_frame, disp_frame, conf_frame)
                        frame_to_save = scene_path_to_save + f
                        self.train_list.append(frame_to_save)

    def preprocess(self):
        """
        Preprocess the dataset

        :return: None
        """
        self._preprocess_monkaa()
        self._preprocess_driving()
        self._preprocess_flyingthings()

        train_txt = os.path.join(self.dataset_path, "train.txt")
        test_txt = os.path.join(self.dataset_path, "test.txt")
        # clear the contents if they are not empty
        open(train_txt, 'w').close()
        open(test_txt, 'w').close()

        with open(train_txt, 'w') as f:
            for entry in self.train_list:
                f.write("%s\n" % entry)

        with open(test_txt, 'w') as f:
            for entry in self.test_list:
                f.write("%s\n" % entry)
