package il.ac.tau.adviplab.myimageproc;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by iplab_4_2 on 3/9/2016.
 */
public class MyImageProc  {

    public static final int HIST_NORMALIZATION_CONST = 10000;
    public static final int COMP_MATCH_DISTANCE = 99;
    public static final int SIGMA_SPATIAL_DEFAULT = 0;
    public static final int SIGMA_INTENSITY_DEFAULT = 0;
    public static final int ALPHA_DEFAULT = 0;
    public static final int BETA_DEFAULT = 0;
    public static final int SIGMA_SPATIAL_MAX = 20;
    public static final int SIGMA_INTENSITY_MAX =100;
    public static final int BETA_MAX = 10;
    public static final int ALPHA_MAX = 1;

    public static void calcHist(Mat image, Mat[] histList, int histSizeNum,
                                int normalizationConst, int normalizationNorm) {
        Mat mat0 = new Mat();
        MatOfInt histSize = new MatOfInt(histSizeNum);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        int numberOfChannel = Math.min(image.channels(), 3);
        MatOfInt[] channels = new MatOfInt[numberOfChannel];
        for (int i = 0; i < numberOfChannel; i++) {
            channels[i] = new MatOfInt(i);
        }

        int chIdx = 0;
        for (MatOfInt channel : channels) {
            Imgproc.calcHist(Arrays.asList(image), channel, mat0,
                    histList[chIdx], histSize, ranges);

            Core.normalize(histList[chIdx], histList[chIdx],
                    normalizationConst, 0, normalizationNorm);
            chIdx++;
        }

        mat0.release();
        histSize.release();
        ranges.release();
        for (MatOfInt channel : channels) {
            channel.release();
        }

    }

    public static void calcHist(Mat image, Mat[] histList, int histSizeNum) {
        int normalizationConst = image.height() / 2;
        int normalizationNorm = Core.NORM_INF;

        calcHist(image, histList, histSizeNum,
                normalizationConst, normalizationNorm);
    }

    public static void showHist(Mat image, Mat[] histList, int histSizeNum,
                                int offset, int thickness) {
        float[] buff = new float[histSizeNum];
        int numberOfChannels = Math.min(image.channels(), 3);
        // if image is RGBA, ignore the last channel
        Point mP1 = new Point();
        Point mP2 = new Point();
        Scalar mColorsRGB[];
        mColorsRGB = new Scalar[]{new Scalar(200, 0, 0, 255), new Scalar(0,
                200, 0, 255), new Scalar(0, 0, 200, 255)};

        for (int i = 0; i < numberOfChannels; i++) {
            Core.normalize(histList[i], histList[i], image.height() / 2, 0, Core.NORM_INF);
        }

        for (int chIdx = 0; chIdx < numberOfChannels; chIdx++) {
            histList[chIdx].get(0, 0, buff);
            for (int h = 0; h < histSizeNum; h++) {
                mP1.x = mP2.x = offset + (chIdx * (histSizeNum + 10) + h) *
                        thickness;
                mP1.y = image.height() - 1;
                mP2.y = mP1.y - 2 - (int) buff[h];
                Core.line(image, mP1, mP2, mColorsRGB[chIdx], thickness);
            }
        }
    }

    public static void showHist(Mat image, Mat[] histList, int histSizeNum) {
        int thickness = Math.min(image.width() / (histSizeNum + 10) / 5, 5);
        int offset = (image.width() - (5 * histSizeNum + 4 * 10) * thickness) / 2;

        showHist(image, histList, histSizeNum, offset, thickness);
    }

    public static void equalizeHist(Mat image) {
        List<Mat> RGBAChannels = new ArrayList<Mat>(4);
        Core.split(image, RGBAChannels);
        for (Mat colorChannel : RGBAChannels) {
            Imgproc.equalizeHist(colorChannel, colorChannel);
        }
        Core.merge(RGBAChannels, image);
        for (Mat mRGBAChannel : RGBAChannels) {
            mRGBAChannel.release();
        }
    }

    public static void calcCumulativeHist(Mat hist, Mat cumuHist) {

        int histSizeNum = (int) hist.total();
        float[] buff = new float[histSizeNum];
        float[] CumulativeSum = new float[histSizeNum];
        hist.get(0, 0, buff);
        float sum = 0;

        for (int h = 0; h < histSizeNum; h++) {
            sum += buff[h];
            CumulativeSum[h] = sum;
        }
        cumuHist.put(0, 0, CumulativeSum);
    }

    public static void calcCumulativeHist(Mat[] hist, Mat[] cumuHist, int numberOfChannels) {
        for (int i = 0; i < numberOfChannels; i++) {
            cumuHist[i].create(hist[i].size(), hist[i].type());
            MyImageProc.calcCumulativeHist(hist[i], cumuHist[i]);
        }
    }


    public static void applyIntensityMapping(Mat srcImage, Mat lookUpTable) {
        Mat tempMat = new Mat();
        Core.LUT(srcImage, lookUpTable, tempMat);
        tempMat.convertTo(srcImage, CvType.CV_8UC1);
        tempMat.release();
    }

    public static void matchHistogram(Mat histSrc, Mat histDst, Mat lookUpTable) {

//        Mat histSrc - source histogram
//        Mat histDst - destination histogram
//        Mat lookUpTable - lookUp table

//        Add your implementation here

        Mat histSrcCum = new Mat(histSrc.size(), histSrc.type());
        Mat histDstCum = new Mat(histDst.size(), histDst.type());

        calcCumulativeHist(histSrc, histSrcCum);
        Core.normalize(histSrcCum, histSrcCum, 1, 0, Core.NORM_INF);
        calcCumulativeHist(histDst, histDstCum);
        Core.normalize(histDstCum, histDstCum, 1, 0, Core.NORM_INF);

        int numOfScales = (int) histSrcCum.total(); //256

        float[] buffSrc = new float[numOfScales];
        float[] buffDst = new float[numOfScales];

        histSrcCum.get(0, 0, buffSrc);
        histDstCum.get(0, 0, buffDst);

        //allocate a buff for the LUT
        int[] buffLUT = new int[numOfScales];
        int j = 0;

        for (int i=0; i< numOfScales; i++){
            while(j< numOfScales && buffSrc[i] > buffDst[j]){
                j++;
            }
            if (j ==256 ) {
                j = 255;
            }
            buffLUT[i] = j;
        }


    lookUpTable.put(0,0,buffLUT);

    //release dynamically allocated memory
    histSrcCum.release();
    histDstCum.release();

}


    public static void matchHist(Mat srcImage, Mat dstImage, Mat[] srcHistArray,
                                 Mat[] dstHistArray, boolean histShow) {
        Mat lookupTable = new Mat(256, 1, CvType.CV_32SC1);
        calcHist(srcImage, srcHistArray, 256);
        compareHistograms(srcImage, srcHistArray[0], dstHistArray[0], new Point(50,50), COMP_MATCH_DISTANCE, "Distance: ");
        matchHistogram(srcHistArray[0], dstHistArray[0], lookupTable);
        applyIntensityMapping(srcImage, lookupTable);
        lookupTable.release();

        if (histShow) {
            Mat[] dstHistArrayForShow = new Mat[3];
            int thickness = Math.min(srcImage.width() / (110) / 5, 5);
            int offset = 2*(srcImage.width()) / 3;
            for (int i = 0; i < dstHistArrayForShow.length; i++) {
                dstHistArrayForShow[i] = new Mat();
            }
            calcHist(dstImage, dstHistArrayForShow, 100);
            showHist(srcImage, dstHistArrayForShow, 100, offset, thickness);

            for (int i = 0; i < dstHistArrayForShow.length; i++) {
                dstHistArrayForShow[i].release();

            }
        }
    }

    public static void compareHistograms(Mat image, Mat h1, Mat h2, Point point, int compType, String string) {
        double dist;
        if (compType == COMP_MATCH_DISTANCE) {
            dist = matchDistance(h1, h2);
        } else {
            dist = Imgproc.compareHist(h1, h2, compType);
        }
        Core.putText(image, string + String.format("%.2f", dist), point, Core.FONT_HERSHEY_COMPLEX_SMALL, 0.8, new Scalar(200, 200, 250), 1);
    }

    public static double matchDistance(Mat h1, Mat h2) {
        double dist = 0;

        Mat cummulativeh1 = new Mat(h1.size(), h1.type());
        Mat cummulativeh2 = new Mat(h2.size(), h2.type());
        Mat diff = new Mat(h2.size(), h2.type());

        //normalize

        calcCumulativeHist(h1, cummulativeh1);
        Core.normalize(h1, h1, 1, 0, Core.NORM_INF);
        calcCumulativeHist(h2, cummulativeh2);
        Core.normalize(h2, h2, 1, 0, Core.NORM_INF);

        Core.absdiff(cummulativeh1, cummulativeh2, diff);
        dist = Core.norm(diff,Core.NORM_L1);

        return dist;
    }

    public static void sobelFilter(Mat inputImage, Mat outputImage, int[]
            window) {
//Applies the Sobel filter to image
        Mat grayInnerWindow = inputImage.submat(window[0], window[1],
                window[2], window[3]);
        Mat grad_x = new Mat();
        Mat grad_y = new Mat();
        int ddepth = CvType.CV_16U;
        Imgproc.Sobel(grayInnerWindow, grad_x, ddepth, 1, 0);
        Core.convertScaleAbs(grad_x, grad_x, 10, 0);
        Imgproc.Sobel(grayInnerWindow, grad_y, ddepth, 0, 1);
        Core.convertScaleAbs(grad_y, grad_y, 10, 0);
        Core.addWeighted(grad_x, 0.5, grad_y, 0.5, 0, outputImage);
        grad_x.release();
        grad_y.release();
    }

    public static void displayFilter(Mat displayImage ,Mat filteredImage,
                                     int[] window) {
        Mat rgbaInnerWindow =
                displayImage.submat(window[0],window[1],window[2],window[3]);
        if (displayImage.channels()>1) {
            Imgproc.cvtColor(filteredImage, rgbaInnerWindow,
                    Imgproc.COLOR_GRAY2BGRA, 4);
        }else{
            filteredImage.copyTo(rgbaInnerWindow);
        }
    }

    public static int[] setWindow(Mat image){
        //Add your implementation here.
        int right   = Math.max(19 * image.width() / 20, 10);
        int left    = Math.max(image.width() / 20,10);
        int top     = Math.max(image.height() / 20,10);
        int bottom =  Math.max(19 * image.height() / 20,10);
        int[] window = new int[]{top, bottom, left, right};// these are theborders of the window
        return window;
    }

    public static void sobelCalcDisplay(Mat displayImage ,Mat inputImage,Mat filteredImage) {
        //The function applies the Sobel filter, and returns the result in a format suitable for display.
        int[] window = setWindow(displayImage);
        sobelFilter(inputImage, filteredImage,window);
        displayFilter(displayImage,filteredImage,window);
    }

    public static void gaussianFilter(Mat inputImage, Mat outputImage,
                                      int[] window, float sigma){
        //Applies gaussian filter to image
        Mat grayInnerWindow =
                inputImage.submat(window[0], window[1], window[2], window[3]);
        Size ksize = new Size(4*(int)sigma+1,4*(int)sigma+1);
        float sigmaX = sigma;
        float sigmaY = sigma;
        try {
            Imgproc.GaussianBlur(grayInnerWindow, outputImage, ksize,
                    sigmaX, sigmaY);
            grayInnerWindow.release();
        }catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    public static void gaussianFilter(Mat inputImage, Mat outputImage,
                                      int[] window){
        gaussianFilter(inputImage, outputImage, window, SIGMA_SPATIAL_DEFAULT);
    }

    public static void gaussianCalcDisplay(Mat displayImage ,Mat
            inputImage,Mat filteredImage, float sigma){

        int[] window = setWindow(displayImage);
        gaussianFilter(inputImage, filteredImage, window, sigma);
        displayFilter(displayImage, filteredImage, window);
    }

    public static void gaussianCalcDisplay(Mat displayImage ,Mat
            inputImage,Mat filteredImage) {

        gaussianCalcDisplay(displayImage, inputImage, filteredImage, SIGMA_SPATIAL_DEFAULT);
    }

    public static void bilateralFilter(Mat inputImage, Mat outputImage,
                                       int[] window, float sigmaSpatial, float sigmaIntensity){
    //Applies bilateralFilter filter to image
        Mat grayInnerWindow =
                inputImage.submat(window[0],window[1],window[2],window[3]);
        int d = 4*(int)sigmaSpatial+1;
        try {
            Imgproc.bilateralFilter(grayInnerWindow, outputImage, d,
                    sigmaIntensity, sigmaSpatial);
            grayInnerWindow.release();
        }catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    public static void bilateralFilter(Mat inputImage, Mat outputImage,
                                       int[] window) {
        bilateralFilter(inputImage, outputImage, window,
                SIGMA_SPATIAL_DEFAULT,
                SIGMA_INTENSITY_DEFAULT);

    }

    public static void bilateralCalcDisplay(Mat displayImage ,Mat
            inputImage,Mat filteredImage){

        bilateralCalcDisplay(displayImage,
                inputImage, filteredImage, SIGMA_SPATIAL_DEFAULT, SIGMA_INTENSITY_DEFAULT);
    }

    public static void bilateralCalcDisplay(Mat displayImage ,Mat
            inputImage,Mat filteredImage,float sigmaSpatial, float sigmaIntensity){

        int[] window = setWindow(displayImage);
        bilateralFilter(inputImage, filteredImage, window, sigmaSpatial, sigmaIntensity);
        displayFilter(displayImage, filteredImage, window);
    }

    public static void unsharpMaskingDisplay(Mat imToDisplay, Mat inputImage, Mat filteredImage, float sigmaSpatial, float alpha,float beta) {
        int[] window = setWindow(imToDisplay);

        inputImage.convertTo(inputImage,CvType.CV_32FC1);
        filteredImage.convertTo(filteredImage, CvType.CV_32FC1);

        Mat grayInnerWindow =
                inputImage.submat(window[0],window[1],window[2],window[3]);

        gaussianFilter(inputImage, filteredImage, window, sigmaSpatial);
        Core.addWeighted(grayInnerWindow,1,filteredImage, -alpha, 0, filteredImage);
        Core.addWeighted(grayInnerWindow,1, filteredImage, beta,0, filteredImage);
        Core.multiply(filteredImage,new Scalar(1/(1+beta-beta*alpha)),filteredImage);

        filteredImage.convertTo(filteredImage, CvType.CV_8UC1);
        displayFilter(imToDisplay, filteredImage, window);
    }

    public static void detecetAndReplaceChessboard(Mat sourceImage, Mat replacementImage){
        int bwThresold = 100;
        Mat bwImage = new Mat();
        MyImageProc.im2BW(sourceImage, sourceImage, bwThresold,
                Imgproc.THRESH_BINARY_INV);

    }
    public static void im2BW(Mat image, Mat bwImage, double threshold, int  type) {
        // type can be either Imgproc.THRESH_BINARY_INV or Imgproc.THRESH_BINARY
        Imgproc.threshold(image, bwImage, threshold, 255 , type);
    }


}