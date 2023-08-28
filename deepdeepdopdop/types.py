from cv2 import Mat
from insightface.app.common import Face as InsightFace

Frame = Mat
Frames = list[Frame]

Face = InsightFace
TargetFaces = list[tuple[int, list[Face]]]
