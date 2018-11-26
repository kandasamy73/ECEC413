Blocked 2D Convolution

3)  Report.
    It's time to do some performance testing and analysis.  Included in the 
    MP3-convolution_block folder is a folder called "test", which contains two 
    test case input sets.  Using these test cases, and any others that you wish 
    to create to support your findings, provide answers to the following questions, 
    along with a short description of how you arrived at those answers.  

    You are free to use any timing library you like, as long as it has a reasonable 
    accuracy.  Note that the CUDA utility library provides some timing functions which 
    are modeled in the given test harness code, it you care to use those.
    Remember that kernel invocations are normally asynchronous, so if you want accurate
    timing of the kernel's running time, you need to insert a call to
    cudaThreadSynchronize() after the kernel invocation.  

    1.  What is the measured floating-point computation rate for the CPU and GPU kernels 
    on this application?  How do they each scale with the size of the input? 

Performance numbers are highly implementation dependent.  However, it is worth
noting that it is impossible to say anything about the shape of a graph with
only two sample points.  A thorough analysis will typically discover that the
GPU computation rate increases with image size, and eventually levels off.
CPU computation rates results tend to level off much more quickly, or even 
fall off as the memory system gets used more poorly by large data sets.  

    2.  How much time is spent as an overhead cost of using the GPU for
    computation?  Consider all code executed within your host function, with
    the exception of the kernel itself, as overhead.  How does the overhead scale 
    with the size of the input?

Again, this is highly implementation dependent, but be aware of whether they
are synchronizing their code for correct timing results.  If it takes an
enormous amount of time to copy back results, but almost nothing to execute
the kernel, you can be almost certain they did not follow the outlined
experimental methodology.  

