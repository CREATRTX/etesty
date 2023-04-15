In particular, this function assumes that np_dir is a subdirectory like one
  created by the latency evaluator for a particular frame, and thus that it
  contains three npy files:
  * boxes.npy: a N x 7 float array with the x, y, z, length, width, height, and
               heading for all the detections in this frame.
  * classes.npy: a N-dim uint8 array with the type IDs in {0, 1, 2, 3, 4} for
                 all the detections in this frame.
  * scores.npy: a N-dim float array with the scores in [0, 1] for all the
                detections in this frame.
  These arrays are converted into a list of N Object protos, one for each
  detection, where all the protos have the frame_context_name and
  frame_timestamp_micros set by the arguments.
  Args:
    np_dir: string directory name containing the npy files.
    frame_context_name: string context_name to set for each Object proto.
    frame_timestamp_micros: int timestamp micros to set for each Object proto.
  Returns:
    List of N Object protos, one for each detection present in the npy files.
    They all have the same context name and frame_timestamp_micros, while their
    boxes, scores, and types come from the numpy arrays.