import cv2
import os 

def get_frames(output_dir, video_file_dir):
    video = cv2.VideoCapture(video_file_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0

    while True:
        ret, frame = video.read()
        if not video.isOpened():
            print("Error: Unable to open video file")
            return
    
        # Break the loop when there are no more frames
        if not ret:
            break
        
        # Increment frame count
        frame_count += 1
        
        # Save the frame with a sequential filename
        filename = os.path.join(output_dir, f"f{frame_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        
        print(f"Frame {frame_count} extracted")
    
    # Release the video capture object
    video.release()

# video_path1 = "IMG_0674.mp4"
video_path2 = "IMG_2372.mp4"
# output_folder1 = "revised_videos/frames/0674_frames"
output_folder2 = "revised_videos/frames/2372_frames"

print("hello?")
# get_frames(output_folder1, video_path1)
get_frames(output_folder2, video_path2)