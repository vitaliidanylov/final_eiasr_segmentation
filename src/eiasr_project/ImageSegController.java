package eiasr_project;

import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

public class ImageSegController
{

	// FXML buttons
	@FXML
	private Button cameraButton;
	// the FXML area for showing the current frame
	@FXML
	private ImageView originalFrame;
	// checkbox for enabling/disabling Watershed
	@FXML
	private CheckBox watershed;
	// checkbox for enabling/disabling Meanshift
	@FXML
	private CheckBox meanshift;
	// watershed threshold value
	@FXML
	private Slider threshold;
	// checkbox for enabling/disabling background removal

	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that performs the video capture
	private VideoCapture capture = new VideoCapture();
	// a flag to change the button behavior
	private boolean cameraActive;

	/**
	 * The action triggered by pushing the button on the GUI
	 */
	@FXML
	protected void startCamera()
	{
		// set a fixed width for the frame
		originalFrame.setFitWidth(380);
		// preserve image ratio
		originalFrame.setPreserveRatio(true);

		if (!this.cameraActive)
		{
			// disable setting checkboxes
			this.watershed.setDisable(true);

			// start the video capture
			this.capture.open(0);

			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;

				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {

					@Override
					public void run()
					{
						Image imageToShow = grabFrame();
						originalFrame.setImage(imageToShow);
					}
				};

				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

				// update the button content
				this.cameraButton.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Failed to open the camera connection...");
			}
		}
		else
		{
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.cameraButton.setText("Start Camera");
			// enable setting checkboxes
			this.watershed.setDisable(false);
			// stop the timer
			try
			{
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log the exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}

			// release the camera
			this.capture.release();
			// clean the frame
			this.originalFrame.setImage(null);
		}
	}

	/**
	 * Get a frame from the opened video stream
	 */
	private Image grabFrame()
	{
		// init everything
		Image imageToShow = null;
		Mat frame = new Mat();

		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);

				// if the frame is not empty, process it
				if (!frame.empty())
				{
					// handle edge detection
					if (this.watershed.isSelected())
					{
						frame = this.doWatershed(frame);
					} else if (this.meanshift.isSelected()){
						frame = this.doMeanshift(frame);
					}
					// convert the Mat object (OpenCV) to Image (JavaFX)
					imageToShow = mat2Image(frame);
				}

			}
			catch (Exception e)
			{
				// log the (full) error
				System.err.print("ERROR");
				e.printStackTrace();
			}
		}

		return imageToShow;
	}



	/**
	 * Apply watershed
	 *
	 * @param frame
	 *            the current frame
	 * @return an image elaborated with watershed
	 */
	private Mat doWatershed(Mat frame)
	{
		//init
		Mat image = frame;
		Mat gray = new Mat();
		Mat bin = new Mat();

		//convert to gray scale and make threshold operation controled by slider
		Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
		Imgproc.threshold(gray, bin, this.threshold.getValue()*2, 255, Imgproc.THRESH_BINARY);

		//init vars for find contours operation
		ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Mat hierarchy = new Mat();

		//detect contours
		Imgproc.findContours(bin, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

		//draw contours as markers for watershed segmentation
		Mat markers = new Mat(image.size(), CvType.CV_32SC1);
		Imgproc.drawContours(markers, contours, -1, new Scalar(255), 1, 8, hierarchy, 1, new Point(0,0));

		//watershed segmentation
		Imgproc.watershed(image, markers);

		return markers;
	}

	/**
	 * Apply meanshift
	 *
	 * @param frame
	 *            the current frame
	 * @return an image elaborated with meanshift
	 */
	private Mat doMeanshift(Mat frame)
	{
		// init
		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat();
		ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Mat hierarchy = new Mat();
		Mat imageSegment = new Mat();

		// constants
		int spatialRadius = 35;
		int colorRadius = 60;

		// convert to grayscale
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);

		// reduce noise with a 3x3 kernel
		Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));

		// Canny detector, with ratio of lower:upper threshold of 3:1
		Imgproc.Canny(detectedEdges, detectedEdges, this.threshold.getValue(), this.threshold.getValue() * 3);

		// find contours
		Imgproc.findContours(detectedEdges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

		//draw contours
		Mat drawing = Mat.zeros(detectedEdges.size(), CvType.CV_8UC3);
		Imgproc.drawContours(drawing, contours, -1, new Scalar(0,255,0));

		//meanshift segmentation
		Imgproc.pyrMeanShiftFiltering(frame, imageSegment, spatialRadius, colorRadius);

		// add images (contours and image after meanShift segmenting)
		Core.add(imageSegment, drawing, imageSegment);

		return imageSegment;
	}

	/**
	 * Action triggered when the watershed checkbox is selected
	 *
	 */
	@FXML
	protected void watershedSelected()
	{
		// enable the threshold slider
		if (this.watershed.isSelected()){
			this.threshold.setDisable(false);
		}
		else
			this.threshold.setDisable(true);
		// now the capture can start
		this.cameraButton.setDisable(false);
	}
	/**
	 * Action triggered when the meanshift checkbox is selected
	 *
	 */
	@FXML
	protected void meanshiftSelected()
	{
		// enable the threshold slider
		if (this.meanshift.isSelected()){
			this.threshold.setDisable(false);
		}
		else
			this.threshold.setDisable(true);

		// now the capture can start
		this.cameraButton.setDisable(false);
	}

	/**
	 * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
	 */
	private Image mat2Image(Mat frame)
	{
		// create a temporary buffer
		MatOfByte buffer = new MatOfByte();
		// encode the frame in the buffer, according to the PNG format
		Imgcodecs.imencode(".png", frame, buffer);
		// build and return an Image created from the image encoded in the
		// buffer
		return new Image(new ByteArrayInputStream(buffer.toArray()));
	}

}
