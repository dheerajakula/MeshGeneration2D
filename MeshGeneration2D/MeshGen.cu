
#include "CUDA_HEADER/QuadTree.cuh"
#include "CUDA_HEADER/delaunay.cuh"
#include "Header/global_datatype.h"

__device__ double2 gpu_voronoi_thresholdpointsforeachedge[MAX_VORONOI_EDGES][MAX_POINTS_SIZE];
__device__ int countof_gpu_voronoi_thresholdpointsforeachedge[MAX_VORONOI_EDGES];
__device__ double2 gpu_nncrust_edgesforeach_voronoithresholdpoint[MAX_VORONOI_EDGES][MAX_POINTS_SIZE * 2];
__device__ double2 gpu_nncrust_intersectionpoints_foreachvoronoi[MAX_VORONOI_EDGES][MAX_POINTS_SIZE * 2];

__device__ double2 gpu_delaunay_edgesforeachvoronoi[MAX_VORONOI_EDGES][6*MAX_POINTS_SIZE - 15];
__device__ int gpu_delaunay_edgesindexforeachvoronoi[MAX_VORONOI_EDGES][6 * MAX_POINTS_SIZE - 15];
__device__ int countof_gpu_delaunay_edgesforeachvoronoi[MAX_VORONOI_EDGES];

Quadtree_Node* d_root;

Points *d_points;

Points* d_inside_points;

double threshold;

void MinMaxCoor(std::vector<Point_2> &input, Iso_rectangle_2& box){
	//double maxX=0,minX=DBL_MAX,maxY=0,minY=DBL_MAX;
	double maxX = 0, minX = DBL_MAX, maxY = 0, minY = DBL_MAX;
	//std::cout.precision(17);
	std::vector<Point_2>::iterator it = input.begin();
	Point_2 point;
	//string str;
	//while((std::getline(input,str))){
	while (it != input.end()){
		point = *it;
		if (minX>point.x()){ minX = (point.x()); }
		if (minY>point.y()){ minY = (point.y()); }
		if (maxX<point.x()){ maxX = (point.x()); }
		if (maxY<point.y()){ maxY = (point.y()); }
		++it;
	}

	maxX = maxX + 100; maxY = maxY + 100; minX = minX - 100; minY = minY - 100;
	Iso_rectangle_2 bbox(minX, minY, maxX, maxY);
	box = bbox;
}

Segment_2 convToSeg(const Iso_rectangle_2& box, const Ray_2& ray){

	CGAL::Object obj = CGAL::intersection(ray, box);

	const Point_2* tempPoint = CGAL::object_cast<Point_2>(&obj);
	const Segment_2* tempSeg = CGAL::object_cast<Segment_2>(&obj);

	if (tempPoint != nullptr){

		Segment_2 temp(ray.source(), *tempPoint);

		return temp;
	}
	if (tempSeg != nullptr){

		return *tempSeg;
	}
}

std::vector<std::pair<int, std::pair<double2,double2>>> process_on_gpu(std::vector<Segment_2> VorEdges, std::vector<Segment_2> DelEdges)
{
	int num_lines = VorEdges.size();

	Line_Segment *h_lines = new Line_Segment[num_lines];
	Line_Segment *d_lines;

	double2* h_delaunayPoints = new double2[num_lines];
	double2* d_delaunayPoints;

	int* h_no_of_intersections = new int[num_lines];
	int* d_no_of_intersections;

	double2* h_intersections = new double2[num_lines];
	double2* d_intersections;

	std::vector<std::pair<int, std::pair<double2, double2>>> ans;


	for (int i = 0; i < VorEdges.size(); i++)
	{
		
		h_lines[i] = Line_Segment(make_double2(VorEdges[i][0].x(), VorEdges[i][0].y()), make_double2(VorEdges[i][1].x(), VorEdges[i][1].y()));
		h_delaunayPoints[i] = make_double2(DelEdges[i][0].x(), DelEdges[i][0].y());
		
	}

	checkCudaErrors(cudaMalloc((void**)&d_lines, num_lines*sizeof(Line_Segment)));
	checkCudaErrors(cudaMemcpy(d_lines, h_lines, num_lines*sizeof(Line_Segment), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&d_delaunayPoints, num_lines*sizeof(double2)));
	checkCudaErrors(cudaMemcpy(d_delaunayPoints, h_delaunayPoints, num_lines*sizeof(double2), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&d_no_of_intersections, num_lines*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_intersections, num_lines*sizeof(double2)));

	findOuterThresholdPoints << <1, 4 >> >(d_root, d_points, d_lines, d_inside_points, threshold);

	delaunayKernel << <1, num_lines >> > (d_lines, d_delaunayPoints, d_no_of_intersections, d_intersections);
	cudaDeviceSynchronize();
	print_delaunay << <1, num_lines >> > ();
	
	cudaError_t cudaStatus;

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	cudaStatus = cudaMemcpy(h_no_of_intersections, d_no_of_intersections, num_lines*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	};

	cudaStatus = cudaMemcpy(h_intersections, d_intersections, num_lines*sizeof(double2), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	};

	
	for (int i = 0; i < num_lines; i++)
	{
		std::pair<double2, double2> p;
		int no_of_intersections = h_no_of_intersections[i];
		
		if (no_of_intersections == 1)
		{
			p = make_pair(make_double2(DelEdges[i][0].x(), DelEdges[i][0].y()), make_double2(DelEdges[i][1].x(), DelEdges[i][1].y()));
		}
		else
		{
			p.first = make_double2(h_intersections[i].x, h_intersections[i].y);
		}
		ans.push_back(make_pair(no_of_intersections, p));
	}


	for (int i = 0; i < num_lines; i++)
	{
		cout << h_no_of_intersections[i] << " " << h_intersections[i].x << " " << h_intersections[i].y << endl;
	}

	delete h_lines;
	delete h_delaunayPoints;
	delete h_no_of_intersections;
	delete h_intersections;

	cudaFree(d_lines);
	cudaFree(d_delaunayPoints);
	cudaFree(d_no_of_intersections);
	cudaFree(d_intersections);

	return ans;
}

int main()
{

	std::string inputFile = "Res/2.5width_4patels.txt";
	// std::string outputFile = "InnerPoints(2.5width_4patels.txt).txt";
	// freopen(outputFile.c_str() , "w", stdout);
	const int max_depth = 10;
	const int min_points_per_node = 5; // Min points per node
	int num_points = -1;

	//Read Points from file and put it into x0(X points) and y0(Y Points)
	std::vector<Point_2> OriginalSample, RandomSample;
	clock_t start = clock();
	std::list<double> stlX, stlY;
	std::ifstream source(inputFile);
	if (source.is_open()){
		int i = 0;
		for (std::string line; std::getline(source, line); i += 1)   //read stream line by line
		{
			std::istringstream in(line);
			double x, y;
			in >> x >> y;
			Point_2 original(x, y);
			OriginalSample.push_back(original);
			stlX.push_back(x);
			stlY.push_back(y);
		}
	}
	else{
		printf("No");
		exit(1);
	}

	/*
	std::ifstream input("neha1.txt");
	int num_of_points = 0;
	std::string data;
	while (getline(input, data))
	{
	Point_2 original;
	std::istringstream stream(data);
	while (stream >> original)
	{
	OriginalSample.push_back(original);
	++num_of_points;
	}
	}
	*/
	clock_t end = clock();
	double run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "File Reading Time: " << run_time << std::endl;
	num_points = stlX.size();
	std::cout << "Number of Points: " << num_points << std::endl;

	Iso_rectangle_2	boundingBox;

	MinMaxCoor(OriginalSample, boundingBox);

	for (int i = 0; i<5; i++)
	{
		int n = std::rand() % (num_points - 1);
		RandomSample.push_back(OriginalSample.at(n));
	}

	Delaunay dt_sample;

	dt_sample.insert(RandomSample.begin(), RandomSample.end());


	//Set Cuda Device
	int device_count = 0, device = -1, warp_size = 0;
	checkCudaErrors(cudaGetDeviceCount(&device_count));
	std::cout << device_count << endl;
	for (int i = 0; i < device_count; ++i)
	{
		cudaDeviceProp properties;
		checkCudaErrors(cudaGetDeviceProperties(&properties, i));
		if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
		{
			device = i;
			warp_size = properties.warpSize;
			std::cout << "Running on GPU: " << i << " (" << properties.name << ")" << std::endl;
			std::cout << "Warp Size: " << warp_size << std::endl;
			std::cout << "Threads Per Block: " << properties.maxThreadsPerBlock << std::endl;
			break;
		}
		std::cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
	}
	if (device == -1)
	{
		//cdpQuadTree requires SM 3.5 or higher to use CUDA Dynamic Parallelism.  Exiting...
		exit(EXIT_SUCCESS);
	}
	cudaSetDevice(device);

	start = clock();
	cudaFree(0);
	cudaThreadSynchronize();
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	end = clock();
	run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "cudaFree Time: " << run_time << std::endl;

	start = clock();
	thrust::device_vector<double> x0(stlX.begin(), stlX.end());
	thrust::device_vector<double> y0(stlY.begin(), stlY.end());
	thrust::device_vector<double> x1(num_points);
	thrust::device_vector<double> y1(num_points);
	end = clock();
	run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "Data Conversion Time: " << run_time << std::endl;

	//copy pointers to the points into the device because kernels don't support device_vector as input they accept raw_pointers
	//Thrust data types are not understood by a CUDA kernel and need to be converted back to its underlying pointer. 
	//host_points(h for host, d for device)

	Points h_points[2];
	h_points[0].set(thrust::raw_pointer_cast(&x0[0]), thrust::raw_pointer_cast(&y0[0]));
	h_points[1].set(thrust::raw_pointer_cast(&x1[0]), thrust::raw_pointer_cast(&y1[0]));


	//device_points
	//Points *d_points;
	checkCudaErrors(cudaMalloc((void**)&d_points, 2 * sizeof(Points)));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(d_points, h_points, 2 * sizeof(Points), cudaMemcpyHostToDevice));
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
	end = clock();
	run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "GPU Data Transfer Time: " << run_time << std::endl;

	//Setting Cuda Heap size for dynamic memory allocation	
	size_t size = 1024 * 1024 * 1024;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);

	//Copy root node from host to device
	Quadtree_Node h_root;
	h_root.setRange(0, num_points);
	h_root.setIdx(1024);
	//Quadtree_Node* d_root;
	checkCudaErrors(cudaMalloc((void**)&d_root, sizeof(Quadtree_Node)));
	checkCudaErrors(cudaMemcpy(d_root, &h_root, sizeof(Quadtree_Node), cudaMemcpyHostToDevice));

	//set the recursion limit based on max_depth
	//maximum possible depth is 24 levels
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);
	Parameters prmtrs(min_points_per_node);
	const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warp_size;
	const int SHARED_MEM_SIZE = 4 * NUM_WARPS_PER_BLOCK*sizeof(int);
	start = clock();
	buildQuadtree << <1, NUM_THREADS_PER_BLOCK, SHARED_MEM_SIZE >> >(d_root, d_points, prmtrs);
	cudaDeviceSynchronize();
	end = clock();
	run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "Kernel Execution Time: " << run_time << std::endl;


	checkCudaErrors(cudaGetLastError());
	printQuadtree << <1, 1 >> >(d_root);
	int num_of_lines = 4;
	printf("Before Inside Initialization\n");
	//Points* d_inside_points = initializeInsidePoints(num_of_lines);
	d_inside_points = initializeInsidePoints(num_of_lines);
	//printf("After Inside points\n");
	cudaDeviceSynchronize();
	Line_Segment *h_lines = new Line_Segment[num_of_lines];
	h_lines[0] = Line_Segment(make_double2(100.0, -200.0), make_double2(0.0, 300.0));
	h_lines[1] = Line_Segment(make_double2(0.0, 300.0), make_double2(600.0, 650.0));
	h_lines[2] = Line_Segment(make_double2(0.0, 300.0), make_double2(-550.0, 680.0));
	h_lines[3] = Line_Segment(make_double2(100.0, -200.0), make_double2(-600.0, -650.0));


	Line_Segment *d_lines;
	checkCudaErrors(cudaMalloc((void**)&d_lines, num_of_lines*sizeof(Line_Segment)));
	checkCudaErrors(cudaMemcpy(d_lines, h_lines, num_of_lines*sizeof(Line_Segment), cudaMemcpyHostToDevice));
	//double threshold = 10;

	double2 *h_delaunayPoints = new double2[num_of_lines];
	h_delaunayPoints[0] = make_double2(10.0,4.0);
	h_delaunayPoints[1] = make_double2(11.0,5.0);
	h_delaunayPoints[2] = make_double2(12.5, 6.0);
	h_delaunayPoints[3] = make_double2(13.2, 7.0);

	double2* d_delaunayPoints;

	checkCudaErrors(cudaMalloc((void**)&d_delaunayPoints, num_of_lines*sizeof(double2)));
	checkCudaErrors(cudaMemcpy(d_delaunayPoints, h_delaunayPoints, num_of_lines*sizeof(double2), cudaMemcpyHostToDevice));

	int *h_no_of_intersections = new int[num_of_lines];
	int* d_no_of_intersections;
	checkCudaErrors(cudaMalloc((void**)&d_no_of_intersections, num_of_lines*sizeof(int)));

	double2 *h_intersections = new double2[num_of_lines];
	double2* d_intersections;
	checkCudaErrors(cudaMalloc((void**)&d_intersections, num_of_lines*sizeof(double2)));
	


	std::cout << "Outer threshold Points: " << std::endl;
	start = clock();
	findOuterThresholdPoints << <1, 4 >> >(d_root, d_points, d_lines, d_inside_points, threshold);
	cudaDeviceSynchronize();
	end = clock();
	run_time = ((double)(end - start) / CLOCKS_PER_SEC);
	std::cout << "Outer threshold Execution Time: " << run_time << std::endl;
	printPoints << <1, 1 >> >(d_inside_points, num_of_lines); // no. of line, points


	printf("____________________________");
	print_gpu_voronoi_thresholdpointsforeachedge << <1, 4 >> >();

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	// Launch a kernel on the GPU with one thread for each element.
	delaunayKernel << <1, 4 >> > (d_lines, d_delaunayPoints, d_no_of_intersections, d_intersections);
	cudaDeviceSynchronize();
	print_delaunay << <1, 4 >> > ();
	
	print_delaunayindex << <1, 4 >> > ();
	cudaDeviceSynchronize();
	print_NNcurst << <1, 4 >> > ();

	

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
	cudaStatus = cudaMemcpy(h_no_of_intersections, d_no_of_intersections, num_of_lines*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	};

	cudaStatus = cudaMemcpy(h_intersections, d_intersections, num_of_lines*sizeof(double2), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	};


	for (int i = 0; i < num_of_lines; i++)
	{
		cout << h_no_of_intersections[i] << " " << h_intersections[i].x <<" "<< h_intersections[i].y << endl;
	}

	std::cout << ".....................................Topology............................." << std::endl;

	std::vector<Segment_2> VorEdges;
	std::vector<Segment_2> DelEdges;
	Segment_2 segm;
	bool iterate = true;

	while (iterate)
	{
	
		Edge_iterator eit = dt_sample.finite_edges_begin();

		for (; eit != dt_sample.finite_edges_end(); ++eit)
		{//2
			if (eit->first->correct_segments[eit->second] == false)
			{
				//std::cout << ".....................................Inside............................." << std::endl;
				//std::cout << eit->first->vertex((eit->second + 1) % 3)->point() << " " << eit->first->vertex((eit->second + 2) % 3)->point() << std::endl;
				iterate = false;

				CGAL::Object o = dt_sample.dual(eit);

				const Segment_2* s = CGAL::object_cast<Segment_2>(&o);
				const Ray_2* r = CGAL::object_cast<Ray_2>(&o);

				int num_of_intersections = 0;
				Segment_2* temp = new Segment_2;
				//ThreshPoints.clear(); NewThreshPoints.clear(); Neighbors.clear(); Neighbor_Segments.clear();

				if (r)
				{
					/*if (tree.rootNode->rectangle.has_on_bounded_side((*r).source())){
					*temp = convToSeg(tree.rootNode->rectangle, *r);
					}*/

					*temp = convToSeg(boundingBox, *r);
				}
				if (s)
				{
					*temp = *s;
				}
				VorEdges.push_back(*temp);
				segm = Segment_2(eit->first->vertex((eit->second + 1) % 3)->point(), eit->first->vertex((eit->second + 2) % 3)->point());
				DelEdges.push_back(segm);
				delete temp;
			}
		}//2

		process_on_gpu(VorEdges, DelEdges);
		for (std::vector<Segment_2>::iterator vi = VorEdges.begin(); vi != VorEdges.end(); ++vi)
		{
			std::cout << "Voronoi Edges		" << vi->source() << " " << vi->end() << std::endl;
		}

		for (std::vector<Segment_2>::iterator di = DelEdges.begin(); di != DelEdges.end(); ++di)
		{
			std::cout << "Delaunay Edges		" << di->source() << " " << di->end() << std::endl;
		}
	}

	int i;
	std::cin >> i;
}