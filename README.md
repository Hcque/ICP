#  Iterative closest point algorithm and its extendtions


The Iterative closest point algorithm is a famous local optimal solution for point cloud registration, applied for a wide spectrum of scenarios such as medicine diagnosis and slam system.
I use Eigen to manipulate the matrix operation. In the following , I will derive the basics.

======================

One drawback for ICP is the bad initial guess will lead to converge locally, such as a hat may matched with a desk corner instead of the hat on the desk.
GoIcp, in "Go-ICP: A Globally Optimal Solution to 
3D ICP Point-Set Registration",  is one remedy for this, It search the solution exhaustively and somehow try to narrow the search space to reduce time compleixty. It use bfs and branch pruning for the rotaion and translation parametric space. I refer to the origin code of the author.

	My work 1. I add the linear distance transform describe in "A GENERAL ALGORITHM FOR COMPUTING DISTANCE 
	TRANSFORMS IN LINEAR TIME", to achieve O(n) construction of Euclidean distance transform, which serves for the O(1) distance compution when finding neigbours.

	My work 2. I make the parallelism by openMP to accelerate the whole algorithm. It is done by parallel for loops at every place if possible. 
	What needs to mention is kdtree, I assign find_neigbour function to separate threads to achieve parallelism.

The experiments data are download from The Stanford 3D Scanning Repository, I try the case bunny rotate 270 degree, the outcome for Go ICP is not satisfactory.
I consider the bound is actually not much narrowed enough to prune branches , then it has to search the whole space.


============================

Another approach is refer to deep learning, "Deep Closest Point: Learning Representations for Point Cloud Registration". It discard the iterative scheme, since the cloest point matching could be viewed as the function from R3 to R3. so why not learn that function use generated data. 
It mapping 3d space points to an  dimension feature space, usually 512 .  The key operation of point cloud is the convolution for its neighbors , here I will omit the details until I has hands-on experience on the dcp. But it indicates my next orientation and exploration.

=============================

I will add two other learning direction of ICP :
1 one method use the particle filter.
2 another is the hardware consideration, since the random memory access of neighbor finding is the bottleneck of a large family of point cloud algorithm.

