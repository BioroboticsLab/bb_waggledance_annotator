# pipeline.py

import numpy as np
import cv2 as cv
from typing import Optional, Tuple


class FramePostprocessingPipeline:
    CROP = 0
    CONTRAST = 1
    MAX_STEPS = 2

    def __init__(self, video_size: Tuple[int, int], screen_size: Tuple[int, int]):
        self.steps = [None] * FramePostprocessingPipeline.MAX_STEPS
        self.options_map = [0] * len(self.steps)

        video_width, video_height = video_size
        screen_width, screen_height = screen_size

        # Calculate aspect ratio of the video
        aspect_ratio = video_width / video_height

        # Set initial window size to video size, unless it's larger than the screen
        if video_width > screen_width or video_height > screen_height:
            # Scale down the video size to fit within screen dimensions, maintaining aspect ratio
            width_ratio = video_width / screen_width
            height_ratio = video_height / screen_height
            if width_ratio > height_ratio:
                new_width = screen_width
                new_height = int(screen_width / aspect_ratio)
            else:
                new_height = screen_height
                new_width = int(screen_height * aspect_ratio)
            self.target_size = (new_width, new_height)
        else:
            self.target_size = (video_width, video_height)

        # Initialize scaling factors based on the initial window size
        self.scale_x = 1.0
        self.scale_y = 1.0

    class PipelineStep:  # this is used as a template.  other classes inherit this, and overwrite these definitions
        def __init__(self, target_size: Tuple[int, int]):
            self.target_size = target_size

        def process(self, frame: np.ndarray) -> np.ndarray:
            return frame

        def transform_coordinates_video_to_screen(
            self, coords: Tuple[int, int]
        ) -> Tuple[int, int]:
            return coords

        def transform_coordinates_screen_to_video(
            self, coords: Tuple[int, int]
        ) -> Tuple[int, int]:
            return coords

    class CropRectAroundCursor(PipelineStep):
        def __init__(self, frame: np.ndarray, mouse_position: Tuple[int, int], target_size: Tuple[int, int]):
            # Initial crop area starts with the entire frame
            self.x = 0
            self.y = 0
            self.w = frame.shape[1]
            self.h = frame.shape[0]     
            super().__init__(target_size)        
            self._target_size = target_size
            self.frame_shape = frame.shape
            self.zoom_factor = 1.0
            self.mouse_position = mouse_position
            self.aspect_ratio = self.frame_shape[1] / self.frame_shape[0]
            self.update_zoom()
            self.update_scaling_factors()

        @property
        def target_size(self):
            return self._target_size

        @target_size.setter
        def target_size(self, value):
            self._target_size = value
            # Update scaling factors when target_size changes
            self.update_scaling_factors()

        def update_scaling_factors(self):
            self.scale_x = self.target_size[0] / self.w
            self.scale_y = self.target_size[1] / self.h

        def update_zoom(self):
            scale = 1.0 / self.zoom_factor

            # Calculate new width based on the zoom factor
            new_w = min(int(scale * self.frame_shape[1]), self.frame_shape[1])
            # Adjust height to maintain the original aspect ratio
            new_h = int(new_w / self.aspect_ratio)

            # Ensure the height doesn't exceed the frame dimensions
            if new_h > self.frame_shape[0]:
                new_h = self.frame_shape[0]
                # Adjust width accordingly to maintain aspect ratio
                new_w = int(new_h * self.aspect_ratio)

            # Mouse position in video coordinates (before zoom is applied)
            mouse_x, mouse_y = self.mouse_position

            # Adjust x and y so that the zoom is centered around the mouse position
            # Calculate the offset of the mouse in the new zoomed view
            new_x = int(mouse_x - (mouse_x - self.x) * new_w / self.w)
            new_y = int(mouse_y - (mouse_y - self.y) * new_h / self.h)

            # Update the width and height to the new zoomed size
            self.w, self.h = new_w, new_h

            # Ensure x and y are within bounds (prevent cropping outside the frame)
            self.x = max(0, min(new_x, self.frame_shape[1] - self.w))
            self.y = max(0, min(new_y, self.frame_shape[0] - self.h))

        def adjust_zoom(self, direction: int) -> bool:
            old_zoom_factor = self.zoom_factor

            if direction > 0:
                self.zoom_factor *= 1.1
            else:
                self.zoom_factor /= 1.1

            self.zoom_factor = max(self.zoom_factor, 1.0)
            self.zoom_factor = min(self.zoom_factor, 8.0)

            # Only update if zoom factor has changed
            if self.zoom_factor != old_zoom_factor:
                # Update the crop rectangle based on the new zoom factor and current mouse position
                self.update_zoom()
                return True
            
            return False

        def update_mouse_position(self, mouse_position: Tuple[int, int]):
            # Update the mouse position for the zoom center
            self.mouse_position = mouse_position

        def process(self, frame: np.ndarray) -> np.ndarray:
            # Crop the frame
            cropped_frame = frame[self.y: self.y + self.h, self.x: self.x + self.w]
            # Resize the cropped frame to the target size
            resized_frame = cv.resize(cropped_frame, self.target_size, interpolation=cv.INTER_LINEAR)
            # Update scaling factors
            self.scale_x = self.target_size[0] / self.w
            self.scale_y = self.target_size[1] / self.h
            return resized_frame

        def transform_coordinates_video_to_screen(
            self, coords: Tuple[int, int]
        ) -> Tuple[int, int]:
            x, y = coords
            # Adjust for cropping and apply scaling
            x_screen = (x - self.x) * self.scale_x
            y_screen = (y - self.y) * self.scale_y
            return int(x_screen), int(y_screen)

        def transform_coordinates_screen_to_video(
            self, coords: Tuple[int, int]
        ) -> Tuple[int, int]:
            x, y = coords
            # Adjust for scaling and cropping in reverse
            x_video = x / self.scale_x + self.x
            y_video = y / self.scale_y + self.y
            return int(x_video), int(y_video)      


    class ContrastNormalizationFast(PipelineStep):
        def __init__(self, frame: np.ndarray):
            H, W = frame.shape[:2]
            mid_y, mid_x = H // 2, W // 2
            crop_h, crop_w = H // 3, W // 3
            center = frame[mid_y - crop_h : mid_y + crop_h, mid_x - crop_w : mid_x + crop_w]

            data = center.flatten()
            self.min, self.max = np.percentile(data, (5, 95))
            self.min = float(self.min)
            self.max = float(self.max) - self.min

        def process(self, frame: np.ndarray) -> np.ndarray:
            frame = frame.astype(np.float32)
            np.subtract(frame, self.min, out=frame)
            np.divide(frame, self.max / 255.0, out=frame)
            np.clip(frame, 0, 255, out=frame)
            frame = frame.astype(np.uint8)
            return frame

    class ContrastNormalizationFastCenter(PipelineStep):
        def __init__(self, frame: np.ndarray):
            H, W = frame.shape[:2]
            mid_y, mid_x = H // 2, W // 2
            crop_h, crop_w = H // 6, W // 6
            center = frame[mid_y - crop_h : mid_y + crop_h, mid_x - crop_w : mid_x + crop_w]

            self.min = center.min()
            self.max = center.max() - self.min
            if self.max <= 0:
                self.max = 1.0

        def process(self, frame: np.ndarray) -> np.ndarray:
            frame = frame.astype(np.float32)
            np.subtract(frame, self.min, out=frame)
            np.divide(frame, self.max / 255.0, out=frame)
            np.clip(frame, 0, 255, out=frame)
            frame = frame.astype(np.uint8)
            return frame

    class ContrastHistogramEqualization(PipelineStep):
        def __init__(self, **kwargs):
            pass

        def process(self, frame: np.ndarray) -> np.ndarray:
            import skimage.exposure

            frame = skimage.exposure.equalize_hist(frame)
            frame = (frame * 255).astype(np.uint8)
            return frame

    def process(self, frame: np.ndarray) -> np.ndarray:
        # Apply each active pipeline step in order
        for step in self.steps:
            if step is not None:
                frame = step.process(frame)
        return frame

    def select_next_option(
        self, option_type: int, options: list, force_index: Optional[int] = None, **kwargs
    ):
        options = [None] + options

        if force_index is None:
            self.options_map[option_type] = (self.options_map[option_type] + 1) % len(options)
        else:
            if self.options_map[option_type] == force_index:
                return
            self.options_map[option_type] = force_index

        new_step_class = options[self.options_map[option_type]]
        self.steps[option_type] = new_step_class(**kwargs) if new_step_class is not None else None

    def select_next_cropping(self, frame: np.ndarray, mouse_position: Tuple[int, int]):
        # Pass the pipeline's target size to the CropRectAroundCursor
        self.steps[FramePostprocessingPipeline.CROP] = FramePostprocessingPipeline.CropRectAroundCursor(
            frame=frame,
            mouse_position=mouse_position,
            target_size=self.target_size  # Pass the inherited target size
        )

    def select_first_cropping(self, **kwargs):
        self.select_next_cropping(force_index=1, **kwargs)

    def disable_cropping(self, **kwargs):
        self.select_next_cropping(force_index=0, **kwargs)

    def select_next_contrast_postprocessing(self, **kwargs):
        options = [
            FramePostprocessingPipeline.ContrastNormalizationFastCenter,
            FramePostprocessingPipeline.ContrastNormalizationFast,
            FramePostprocessingPipeline.ContrastHistogramEqualization,
        ]
        self.select_next_option(FramePostprocessingPipeline.CONTRAST, options, **kwargs)

    def transform_coordinates_video_to_screen(
        self, coords: Tuple[int, int]
    ) -> Tuple[int, int]:
        x, y = coords
        # Apply transformations from pipeline steps
        for step in self.steps:
            if step is not None and hasattr(step, 'transform_coordinates_video_to_screen'):
                x, y = step.transform_coordinates_video_to_screen((x, y))
        # No additional scaling needed if resizing is handled in steps
        return int(x), int(y)

    def transform_coordinates_screen_to_video(
        self, coords: Tuple[int, int]
    ) -> Tuple[int, int]:
        x, y = coords
        # Apply inverse transformations from pipeline steps
        for step in reversed(self.steps):
            if step is not None and hasattr(step, 'transform_coordinates_screen_to_video'):
                x, y = step.transform_coordinates_screen_to_video((x, y))
        # No additional scaling needed if resizing is handled in steps
        return int(x), int(y)  

    def adjust_zoom(self, direction: int) -> bool:
        was_handled = False
        step = self.steps[FramePostprocessingPipeline.CROP]
        if step is not None:
            handled_now = step.adjust_zoom(direction)
            was_handled = was_handled or handled_now

        return was_handled
    
    def update_mouse_position(self, mouse_position: Tuple[int, int]):
        step = self.steps[FramePostprocessingPipeline.CROP]
        if step is not None:
            step.update_mouse_position(mouse_position)    