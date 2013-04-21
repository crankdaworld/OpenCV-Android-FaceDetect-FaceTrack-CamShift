package org.opencv.samples.fd;

import java.util.List;
import java.util.Vector;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;

public class TrackedObj
{
	Mat hsv,hue,mask,prob;
	Rect prev_rect;
	RotatedRect  curr_box;
	//http://stackoverflow.com/questions/10324727/opencv-how-to-convert-a-cpp-code-in-c
	Mat hist;
	public List<Mat> hsvarray,huearray;

	public TrackedObj()
	{
	//hsv=new Mat();
	//hue=new Mat();
	//mask=new Mat();
	//prob=new Mat();
	hist=new Mat();
	prev_rect=new Rect();
	curr_box=new RotatedRect();
	hsvarray=new Vector<Mat>();
	huearray=new Vector<Mat>();
	
	}
}
