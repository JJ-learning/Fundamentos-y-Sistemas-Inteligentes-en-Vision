#include <iostream>
#include <exception>
#include <ctype.h>
#include <cmath>
#include <sstream>
#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/highgui.hpp>		   // imread
#include <opencv2/imgproc/imgproc.hpp> // cvtcolor
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int alpha_slider_max = 255;
int alpha_slider;
int alpha;

const int beta_slider_max = 20;
int beta_slider;
int beta;

void on_trackbar(int, void *)
{
	alpha = alpha_slider;
}

void on_trackbar2(int, void *)
{
	if (beta_slider == 0)
	{
		beta_slider++;
	}
	beta = beta_slider;
}

bool has_only_digits(string path)
{
	return path.find_first_not_of("0123456789") == string::npos;
}

const String keys =
	"{help h usage ? || print this message\n}"
	"{@path |.| path to file\n}"
	"{@out |out.png| path for the output image\n}"
	"{s || size \n}"
	"{t || threshold\n}";

int main(int argc, char *const *argv)
{
	Mat frame;
	bool primero = true;
	int umbral = 0;

	int retCode = EXIT_SUCCESS;

	try
	{
		CommandLineParser parser(argc, argv, keys);
		parser.about("Application name v1.0.0");
		if (parser.has("help"))
		{
			parser.printMessage();
			return 0;
		}

		String input_video = parser.get<String>(0);
		String output = parser.get<String>(1);
		float umbral = parser.get<float>("t");
		int radio = parser.get<int>("s");

		if (!parser.check())
		{
			parser.printErrors();
			return 0;
		}

		if (umbral < 0 || radio <= 0)
		{
			cout << "Error Parametros" << endl;
			return 0;
		}

		int check;
		uint numFrame = 0;
		umbral = 0;
		beta = 0;
		beta_slider = 0;
		alpha = 0;

		VideoCapture video;
		if (has_only_digits(input_video))
		{
			video.open(0);
		}
		else
		{
			video.open(input_video);
		}

		string guardarImagen;

		Mat primerPlano;
		Mat mascara;
		Mat diferencia;

		vector<Mat> canalesClean;

		double fps = 25.0;
		fps = video.get(CV_CAP_PROP_FPS);
		int frame_width = video.get(CV_CAP_PROP_FRAME_WIDTH);
		int frame_height = video.get(CV_CAP_PROP_FRAME_HEIGHT);

		VideoWriter videoSalida("output.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, Size(frame_width, frame_height), true);

		int ex = static_cast<int>(video.get(CV_CAP_PROP_FOURCC)); // Get Codec Type- Int form
		Size S = Size((int)video.get(CV_CAP_PROP_FRAME_WIDTH),	// Acquire input_video size
					  (int)video.get(CV_CAP_PROP_FRAME_HEIGHT));

		if (parser.has("t"))
		{
			umbral = umbral;
			alpha = umbral;
			alpha_slider = umbral;
		}

		if (parser.has("s"))
		{
			beta = radio;
			beta_slider = radio;
		}

		if (!video.isOpened())
		{
			cout << "No se ha podido abrir el video. Compruebe que la ruta es correcta. " << endl;
			exit(-1);
		}
		else
		{
			namedWindow(input_video, 1);

			createTrackbar("Umbral", input_video, &alpha_slider, alpha_slider_max, on_trackbar);
			createTrackbar("Radio", input_video, &beta_slider, beta_slider_max, on_trackbar2);

			int fps = 30;
			videoSalida.open(output, ex, fps, S, true);

			if (!videoSalida.isOpened())
			{
				cout << " No se ha podido crear el video de salida." << endl;
				exit(-1);
			}

			while (video.grab())
			{
				Mat frame1, frame2;
				Mat opening;
				Mat closing;

				video.retrieve(frame1);
				umbral = alpha;
				if (video.read(frame))
				{
					video.retrieve(frame2);
					// Calculamos la diferencia
					absdiff(frame1, frame2, diferencia);
					diferencia = diferencia > umbral;

					if (radio > 0)
					{
						radio = beta;
						Mat kern = getStructuringElement(2, Size(radio * 2 + 1, radio * 2 + 1), Point(radio, radio));
						// Realizamos el opening
						morphologyEx(diferencia, opening, 2, kern);
						// Realizamos el closing
						morphologyEx(opening, closing, 3, kern);

						closing = frame1 & closing;
					}

					videoSalida << closing;
					imshow(input_video, closing);

					check = (char)waitKey(50);
					// Si el usuario pulsa espacio
					if (check == 32)
					{
						cout << "Frame Guardado" << endl;
						stringstream sstm;
						string name = "sal_";

						sstm << numFrame;
						name = name + sstm.str();
						name = name + ".png";

						imwrite(name, closing);
					}
					// Si el usuario pulsa escape
					if (check == 27)
					{
						destroyAllWindows();
						exit(1);
					}
					// Aumentamos le numero de frame para guardarlos
					numFrame++;
				}
				else
				{
					cout << "Se acaba el video." << endl;
					destroyAllWindows();
					exit(1);
				}
			}
		}
	}
	catch (exception &e)
	{
		cerr << "Capturada excepcion: " << e.what() << endl;
		retCode = EXIT_FAILURE;
	}
	return retCode;
}
