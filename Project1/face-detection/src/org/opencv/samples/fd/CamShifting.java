/**
* Copyright (C) 2013 Imran Akthar (www.imranakthar.com)
* imran@imranakthar.com
*/

package org.opencv.samples.fd;

import java.util.Arrays;
import java.util.List;
import java.util.Vector;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;

import android.util.Log;
//http://stackoverflow.com/questions/9804254/image-comparison-of-logos
//http://opencv.willowgarage.com/documentation/histograms.html
//http://code.opencv.org/issues/1447
//http://answers.opencv.org/question/3650/trying-to-calculate-histogram-on-android-and-find/
//http://answers.opencv.org/question/664/camshift-in-android/
//https://groups.google.com/forum/?fromgroups=#!topicsearchin/android-opencv/calcHist/android-opencv/v_TNQea3xxM
//http://grokbase.com/t/gg/android-opencv/122g44w1vp/histogram-calculation
//http://android-spikes.googlecode.com/svn/HelloImageDetection/src/gestoreImmagini/HistogramCompareUtil.java
//https://projects.developer.nokia.com/opencv/browser/opencv/opencv-2.3.1/modules/java/android_test/src/org/opencv/test/imgproc/ImgprocTest.java?rev=ffd62ba23055b3d4b8ba068d5554e2760f0f0eea
//https://code.ros.org/trac/opencv/changeset/6111/trunk/opencv/modules/java/android_test/src/org/opencv/test/imgproc/imgprocTest.java


public class CamShifting
{
	private TrackedObj obj;
	int hist_bins;           //number of histogram bins
    int hist_range[]= {0,180};//histogram range
    int range;
    Mat bgr;
  

	public CamShifting()
	{
	obj=new TrackedObj();
	//hist_range[]= {0,180};
	hist_bins=30;
	//bgr=new Mat();
	
	//range=hist_range;
	}
	
	public void create_tracked_object(Mat mRgba,Rect[] region,CamShifting cs)
	{
		 cs.obj.hsv=new Mat(mRgba.size(),CvType.CV_8UC3);
		  // obj->hsv  = cvCreateImage(cvGetSize(image), 8, 3);
		  cs.obj.mask=new Mat(mRgba.size(),CvType.CV_8UC1);
		  //obj->mask = cvCreateImage(cvGetSize(image), 8, 1);
		  cs.obj.hue=new Mat(mRgba.size(),CvType.CV_8UC1);
		  //obj->hue  = cvCreateImage(cvGetSize(image), 8, 1);
		  cs.obj.prob=new Mat(mRgba.size(),CvType.CV_8UC1);
		  //obj->prob = cvCreateImage(cvGetSize(image), 8, 1);
		update_hue_image(mRgba,region,cs);
		
		
		float max_val = 0.f;
		
		//create a histogram representation for the face
		//Rect roi = new Rect((int)region[0].tl().x,(int)(region[0].tl().y),region[0].width,region[0].height);//imran 
		  Mat tempmask=new Mat(cs.obj.mask.size(),CvType.CV_8UC1);			 
		  //tempmask=cs.obj.mask.submat(roi);
		  tempmask=cs.obj.mask.submat(region[0]);
		  
		  
		 // Log.i("CamShifting","Mask Size"+tempmask.size());
		  //cant use mask here as method wil not take
		  MatOfFloat ranges = new MatOfFloat(0f, 256f);
		  MatOfInt histSize = new MatOfInt(25);
		  //List<Mat> histList = Arrays.asList( new Mat[] {new Mat(), new Mat(), new Mat()} );
		 // Imgproc.calcHist(cs.obj.huearray, new MatOfInt(0),cs.obj.mask, cs.obj.hist, histSize, ranges);
		 // List<Mat> images = Arrays.asList(cs.obj.hsv.submat(roi));
		  List<Mat> images = Arrays.asList(cs.obj.huearray.get(0).submat(region[0]));
		  Imgproc.calcHist(images, new MatOfInt(0),tempmask, cs.obj.hist, histSize, ranges);
		  
		  Core.normalize(cs.obj.hist, cs.obj.hist);
		  //Core.normalize(cs.obj.hist, cs.obj.hist, 0,255,Core.NORM_MINMAX);
		  cs.obj.prev_rect=region[0];
		  Log.i("Normalized Histogram","Normalized Histogram Starting"+cs.obj.hist);
		
		
	}
	
	public void update_hue_image(Mat mRgba,Rect[] region,CamShifting cs)
	{
		
		  int vmin =65, vmax = 256, smin = 55;
		  bgr=new Mat(mRgba.size(),CvType.CV_8UC3);
		  Imgproc.cvtColor(mRgba,bgr,Imgproc.COLOR_RGBA2BGR);
		  //imran converting RGBA to BGR 
		//convert to HSV color model
		  Imgproc.cvtColor(bgr,cs.obj.hsv,Imgproc.COLOR_BGR2HSV);
		  
		//mask out-of-range values
		  Core.inRange(cs.obj.hsv, new Scalar(0, smin,Math.min(vmin,vmax)),new Scalar(180, 256,Math.max(vmin, vmax)), cs.obj.mask);
		  
		  cs.obj.hsvarray.clear();
		  cs.obj.huearray.clear();
		  cs.obj.hsvarray.add(cs.obj.hsv);
		  cs.obj.huearray.add(cs.obj.hue);
		  MatOfInt from_to = new MatOfInt(0,0);
		  //extract the hue channel, split: src, dest channels
		  Core.mixChannels(cs.obj.hsvarray,cs.obj.huearray,from_to);
	}
	
	RotatedRect camshift_track_face(Mat mRgba,Rect[] region,CamShifting cs)
	{
		
		MatOfFloat ranges = new MatOfFloat(0f, 256f);
		//ConnectedComp components;
		update_hue_image(mRgba,region,cs);
		Imgproc.calcBackProject(cs.obj.huearray, new MatOfInt(0),cs.obj.hist,cs.obj.prob, ranges,255);
		Core.bitwise_and(cs.obj.prob,cs.obj.mask,cs.obj.prob,new Mat());
		
		cs.obj.curr_box=Video.CamShift(cs.obj.prob, cs.obj.prev_rect, new TermCriteria(TermCriteria.EPS,10,1));	
		Log.i("Tracked Rectangle","Tracked Rectangle"+cs.obj.prev_rect);
		Log.i("Tracked Rectangle","New Rectangle"+cs.obj.curr_box.boundingRect());
		cs.obj.prev_rect=cs.obj.curr_box.boundingRect();
		cs.obj.curr_box.angle=-cs.obj.curr_box.angle;
		return cs.obj.curr_box;
	}
	
	
	
}
