
typedef struct PoseEstimator {
	PEModel *model ;
	PEParameters *params ;
	PEHyperParameters *hparams ;
} PoseEstimator ;

typedef struct TrainingHyperParameters {
	int maxEpochs ;
	enum optimizationType ;
	// ... much other stuff ...
} TrainingHyperParameters;

typedef struct Pose {
	// ...
	// comprises lists of returned poses, 
	//	plus additional info (errors, etc.)
	//
} Pose ;

PoseEstimator *initPoseEstimator(
	PEParameters *params, PEHyperParameters *hparams) ;

PoseEstimator *trainPoseEstimator(
	PoseEstimator *poseEstimator,
	TrainingHyperParameters *trainingHyperParameters) ;

Pose *estimatePose(
	Image *image, 
	PoseEstimator *poseEstimator) ;
