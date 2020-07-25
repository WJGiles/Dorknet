import cv2, os, functools, itertools, random, numpy as np
from multiprocessing.pool import ThreadPool
import threading
from pprint import pprint
import queue

import time

class ImageDataLoader:
    def __init__(self, base_folder, batch_size, preprocessor,
                 classes_from_dir_structure=True, num_workers=1,
                 class_balance=True, mixup_range_tuple=None, 
                 start_thread=True):
        self.keep_loading = True
        self.preprocessor = preprocessor
        self.mixup_range_tuple = mixup_range_tuple
        self.base_folder = base_folder
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.batch_queue = queue.Queue(maxsize=5)
        self.class_balance = class_balance
        self.queue_thread = threading.Thread(target=self.load_batch,
                                             args=(self.class_balance,))
        self.pause_message_queue = queue.Queue(maxsize=1)
        self.restart_message_queue = queue.Queue(maxsize=1)
        if classes_from_dir_structure:
            self.class_names = [c for c in os.listdir(base_folder)
                                  if os.path.isdir(os.path.join(base_folder, c))]
            self.class_name_num_map = {name: num for num, name in enumerate(sorted(self.class_names))}
            class_name_to_image_paths_map = {c_n: [os.path.join(base_folder, c_n, "images", f)
                                            for f in os.listdir(os.path.join(base_folder, c_n, "images"))]
                                            for c_n in self.class_names}
            self.class_name_to_image_iterator_map = {c_n: itertools.cycle(p) 
                                                        for c_n, p in class_name_to_image_paths_map.items()}
            self.class_cycle = itertools.cycle(self.class_name_to_image_iterator_map.items())
            self.class_name_path_tuple_list = [(c, p) for c in class_name_to_image_paths_map.keys()
                                                 for p in class_name_to_image_paths_map[c]]
            self.index_cycle = itertools.cycle(range(len(self.class_name_path_tuple_list)))
            print("Number of samples: ", len(self.class_name_path_tuple_list))
        if start_thread:
            self.queue_thread.start()

    def stop_thread(self):
        self.keep_loading = False
        # Grab a batch in case load_batch is blocking on a put
        X_batch, y_batch, y_one_hot = self.batch_queue.get()
        # If we are using mixup, batches get added in pairs so need to pull 2
        if self.mixup_range_tuple is not None:
            X_batch, y_batch, y_one_hot = self.batch_queue.get()
        self.queue_thread.join()

    def shuffle_indices(self):
        # Need to ask batch_loader to pause, 
        # clear loaded batches, shuffle, 
        # then start again
        self.pause_message_queue.put("Wait please")
        # Grab a batch in case load_batch is blocking on a put
        X_batch, y_batch, y_one_hot = self.batch_queue.get()
        # If we are using mixup, batches get added in pairs so need to pull 2
        if self.mixup_range_tuple is not None:
            X_batch, y_batch, y_one_hot = self.batch_queue.get()
        self.pause_message_queue.join()
        current_len = self.batch_queue.qsize()
        for i in range(current_len):
            X_batch, y_batch, y_one_hot = self.batch_queue.get()
        self.index_cycle = itertools.cycle(
            list(np.random.permutation(len(self.class_name_path_tuple_list)))
        )
        self.restart_message_queue.put("Start please")

    def get_batch_list(self, class_balance=True):
        while True:
            X_batch_list = []
            y_batch_list = []
            if class_balance:
                for i in range(self.batch_size):
                    c_name, path_cycle = next(self.class_cycle)
                    y_batch_list.append(self.class_name_num_map[c_name])
                    X_batch_list.append(next(path_cycle))
            else:
                for i in range(self.batch_size):
                    c_name, path = self.class_name_path_tuple_list[next(self.index_cycle)]
                    y_batch_list.append(self.class_name_num_map[c_name])
                    X_batch_list.append(path)

            yield X_batch_list, y_batch_list

    def load_batch(self, class_balance):
        keep_going = True
        while self.keep_loading:
            if not self.pause_message_queue.empty():
                message = self.pause_message_queue.get()
                keep_going = False
                self.pause_message_queue.task_done()
            if keep_going:
                X_batch_list, y_batch_list = next(self.get_batch_list(class_balance=class_balance))
                with ThreadPool(self.num_workers) as p:
                    X_batch = np.stack(p.map(self.preprocessor.load_image, X_batch_list), axis=0)
                one_hot_y = np.array([np.eye(len(self.class_names), dtype=np.float32)[i, :] for i in y_batch_list])
                if self.mixup_range_tuple is not None:
                    # Create 'mixed up' batches as convex combination
                    mixup_prop = np.random.uniform(*self.mixup_range_tuple)
                    X_batch_list_m, y_batch_list_m = next(self.get_batch_list(class_balance=class_balance))
                    with ThreadPool(self.num_workers) as p:
                        X_batch_m = np.stack(p.map(self.preprocessor.load_image, X_batch_list_m), axis=0)
                    one_hot_y_m = np.array([np.eye(len(self.class_names), dtype=np.float32)[i, :] for i in y_batch_list_m])
                    X_batch_mixed = mixup_prop*X_batch_m + (1 - mixup_prop)*X_batch
                    X_batch_mixed_m = mixup_prop*X_batch + (1 - mixup_prop)*X_batch_m
                    one_hot_y_mixed = mixup_prop*one_hot_y_m + (1 - mixup_prop)*one_hot_y
                    one_hot_y_mixed_m = mixup_prop*one_hot_y + (1 - mixup_prop)*one_hot_y_m
                    self.batch_queue.put((X_batch_mixed, y_batch_list, one_hot_y_mixed))
                    self.batch_queue.put((X_batch_mixed_m, y_batch_list_m, one_hot_y_mixed_m))
                else:
                    self.batch_queue.put((X_batch, y_batch_list, one_hot_y))
            else:
                message = self.restart_message_queue.get()
                keep_going = True

    def pull_batch(self, num_steps):
        for i in range(num_steps):
            X_batch, y_batch, y_one_hot = self.batch_queue.get()
            yield X_batch, y_batch, y_one_hot
