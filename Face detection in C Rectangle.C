#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>

static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;

void detect_and_draw( IplImage* image );

const char* cascade_name = "haarcascade_frontalface_alt.xml";


int main( int argc, char** argv ) {

    CvCapture* camera = 0; //Initialize webcam
    IplImage *frame, *frame_copy = 0;


    cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 ); //load haarcascade

    if( !cascade )
    {
        printf( "ERROR: Could not load classifier cascade\n" );
        return -1;
    }
    storage = cvCreateMemStorage(0);
    camera = cvCreateCameraCapture( 0 );
    cvNamedWindow( "result", 1 );


    if( camera )
    {
         while ( 1 ) {
	// Grab the next frame from the camera.
		frame = cvQueryFrame(camera);

		if (!frame) {
			printf("Couldn't grab a camera frame.\n");
			break;
		}


    frame_copy = cvCloneImage(frame);
    // Call the function to detect and draw the face
    detect_and_draw( frame_copy );
	char c = (char)cvWaitKey( 10 );
	if( c == 27 || c == 'q' || c == 'Q' ) //exit on esc key or q or Q
		break;
	}
	cvReleaseImage( &frame_copy );
    cvReleaseCapture( &camera );
        }

    else
        printf("Couldn't open camera...");


    cvDestroyWindow("result");
    cvReleaseHaarClassifierCascade( &cascade );
    cvReleaseMemStorage( &storage );

    return 0;
}

void detect_and_draw( IplImage* img ) {

    int nfaces;
    static CvScalar colors[] =
    {
        {{0,0,255}},
        {{0,128,255}},
        {{0,255,255}},
        {{0,255,0}},
        {{255,128,0}},
        {{255,255,0}},
        {{255,0,0}},
        {{255,0,255}}
    };

    double scale = 1.3;
    IplImage* gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
    IplImage* small_img = cvCreateImage( cvSize( cvRound (img->width/scale),cvRound (img->height/scale)),8, 1 );
    int i;

    cvCvtColor( img, gray, CV_BGR2GRAY );
    cvResize( gray, small_img, CV_INTER_LINEAR );
    cvEqualizeHist( small_img, small_img );
    cvClearMemStorage( storage );

    if( cascade )
    {
        double t = (double)cvGetTickCount();
        CvSeq* faces = cvHaarDetectObjects( small_img, cascade, storage,1.1, 2, 0|CV_HAAR_SCALE_IMAGE,cvSize(20, 20),cvSize(250,250) );
        t = (double)cvGetTickCount() - t;
        nfaces = faces->total;
        printf( "%d faces detected in %gms\n",nfaces, t/((double)cvGetTickFrequency()*1000.) );
        for( i = 0; i < (faces ? faces->total : 0); i++ )
        {
            CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
            CvPoint pt1;
            CvPoint pt2;
            //plot the points on the face border
            pt1.x = r->x*scale;
            pt2.x = (r->x+r->width)*scale;
            pt1.y = r->y*scale;
            pt2.y = (r->y+r->height)*scale;

            // Draw the rectangle in the input image
            cvRectangle( img, pt1, pt2, colors[i%8], 3, 8, 0 );

        }
    }

    cvShowImage( "result", img );
    cvReleaseImage( &gray );
    cvReleaseImage( &small_img );
}
