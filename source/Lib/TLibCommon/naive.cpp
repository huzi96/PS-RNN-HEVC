#include "naive.h"
extern Mem mm;


void free_buffer(void* data, size_t length) {
  free(data);
}

void deallocator(void* ptr, size_t len, void* arg) {
  free((void*) ptr);
}

TF_Buffer* TF_Predict::read_file(const char* file) {
    FILE *f = fopen(file, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);  //same as rewind(f);

    void* data = malloc(fsize);
    fread(data, fsize, 1, f);
    fclose(f);

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = free_buffer;
    return buf;
}
int TF_Predict::init(string model_file, string trained_file, int _width)
{
    //Trained file is no longer needed in this function
    // load graph
    // ================================================================================
    width = _width;
    TF_Buffer* graph_def = read_file(model_file.c_str());
    graph = TF_NewGraph();
    status = TF_NewStatus();
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to import graph %s\n", TF_Message(status));
        return 1;
    }
    fprintf(stdout, "Successfully imported graph\n");


    // create session
    // ================================================================================
    TF_SessionOptions* opt = TF_NewSessionOptions();
    char s[] = {0x10, 0x1, 0x28, 0x1, 0x32, 0x2, 0x20, 0x1, 0x52, 0x4, 0x1a, 0x2, 0x28, 0x1};
    TF_SetConfig(opt, s, 14, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Failed setting parallelism %s\n", TF_Message(status));
        return 1;
    }
    sess = TF_NewSession(graph, opt, status);
    TF_DeleteSessionOptions(opt);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to create session %s\n", TF_Message(status));
        return 1;
    }
    fprintf(stdout, "Successfully created session\n");

    inbound = new Hmat<unsigned short>(width * 3, width * 3);
    outbound = new Hmat<unsigned short>(width, width);

}

int TF_Predict::predict()
{
    int i,j;
    // gerenate input
    // ================================================================================
    TF_Operation *input_op = TF_GraphOperationByName(graph, "Placeholder");
    // printf("input_op has %i inputs\n", TF_OperationNumOutputs(input_op));
    float* raw_input_data = (float*)malloc(width*3*width*3 * sizeof(float));
    for(i = 0; i < width*3; i++)
    {
        for(j = 0; j < width*3; j++)
        {
            raw_input_data[j+i*width*3] = (*inbound)[i][j]/mm.norm_value;
        }
    }
    int64_t* raw_input_dims = (int64_t*)malloc(4 * sizeof(int64_t));
    raw_input_dims[0] = 1;
    raw_input_dims[1] = width*3;
    raw_input_dims[2] = width*3;
    raw_input_dims[3] = 1;

    /*
    TF_CAPI_EXPORT extern TF_Tensor* TF_NewTensor(
      TF_DataType,
      const int64_t* dims, int num_dims,
      void* data, size_t len,
      void (*deallocator)(void* data, size_t len, void* arg),
      void* deallocator_arg);
    */
    // prepare inputs
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT,
                                         raw_input_dims, 4,
                                         raw_input_data, width*3*width*3 * sizeof(float),
                                         deallocator,
                                         NULL
                                        );


    // void* input_data = TF_TensorData(input_tensor);
    // printf("input_data[0] = %f\n", ((float*)input_data)[0]);
    // printf("input_data[1] = %f\n", ((float*)input_data)[1]);


    TF_Output* run_inputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
    run_inputs[0].oper = input_op;
    run_inputs[0].index = 0;

    TF_Tensor** run_inputs_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
    run_inputs_tensors[0] = input_tensor;

    // prepare outputs
    // ================================================================================
    TF_Operation* output_op;
    if (width >= 16)
    {
        output_op = TF_GraphOperationByName(graph, "main_full/conv11/BiasAdd");
        // output_op = TF_GraphOperationByName(graph, "main_full/postprocess_conv11/BiasAdd");
    }
    else
    {
        output_op = TF_GraphOperationByName(graph, "main_full/conv11/BiasAdd");
    }
    // printf("output_op has %i outputs\n", TF_OperationNumOutputs(output_op));

    TF_Output* run_outputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
    run_outputs[0].oper = output_op;
    run_outputs[0].index = 0;


    TF_Tensor** run_output_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
    float* raw_output_data = (float*)malloc(width*width * sizeof(float));
    raw_output_data[0] = 1.f;
    int64_t* raw_output_dims = (int64_t*)malloc(4 * sizeof(int64_t));
    raw_output_dims[0] = 1;
    raw_output_dims[1] = width;
    raw_output_dims[2] = width;
    raw_output_dims[3] = 1;

    TF_Tensor* output_tensor = TF_NewTensor(TF_FLOAT,
                                          raw_output_dims, 4,
                                          raw_output_data, width*width * sizeof(float),
                                          deallocator,
                                          NULL
                                         );
    run_output_tensors[0] = output_tensor;

    // run network
    // ================================================================================
    TF_SessionRun(sess,
                /* RunOptions */         NULL,
                /* Input tensors */      run_inputs, run_inputs_tensors, 1,
                /* Output tensors */     run_outputs, run_output_tensors, 1,
                /* Target operations */  NULL, 0,
                /* RunMetadata */        NULL,
                /* Output status */      status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to run output_op: %s\n", TF_Message(status));
        return 1;
    }

    // printf("output-tensor has %i dims\n", TF_NumDims(run_output_tensors[0]));

    void* output_data = TF_TensorData(run_output_tensors[0]);
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            (*outbound)[i][j] = (*outbound)[i][j] = (unsigned short)round(((float*)output_data)[j+i*width] * mm.norm_value);
        }
    }

    // free((void*) raw_input_data);
    // free((void*) raw_output_data);
    free((void*) run_inputs);
    free((void*) run_outputs);
    free((void*) run_inputs_tensors);
    free((void*) run_output_tensors);
    free((void*) raw_input_dims);
    free((void*) raw_output_dims);
}


TF_Buffer* TF_HalfPredict::read_file(const char* file) {
    FILE *f = fopen(file, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);  //same as rewind(f);

    void* data = malloc(fsize);
    fread(data, fsize, 1, f);
    fclose(f);

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = free_buffer;
    return buf;
}
int TF_HalfPredict::init(string model_file, string trained_file, int _width)
{
    //Trained file is no longer needed in this function
    // load graph
    // ================================================================================
    width = _width;
    TF_Buffer* graph_def = read_file(model_file.c_str());
    graph = TF_NewGraph();
    status = TF_NewStatus();
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, graph_def, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to import graph %s\n", TF_Message(status));
        return 1;
    }
    fprintf(stdout, "Successfully imported graph\n");


    // create session
    // ================================================================================
    TF_SessionOptions* opt = TF_NewSessionOptions();
    char s[] = {0x10, 0x1, 0x28, 0x1, 0x32, 0x2, 0x20, 0x1, 0x52, 0x4, 0x1a, 0x2, 0x28, 0x1};
    TF_SetConfig(opt, s, 14, status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Failed setting parallelism %s\n", TF_Message(status));
        return 1;
    }
    sess = TF_NewSession(graph, opt, status);
    TF_DeleteSessionOptions(opt);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to create session %s\n", TF_Message(status));
        return 1;
    }
    fprintf(stdout, "Successfully created session\n");

    inbound = new Hmat<unsigned short>(width * 2, width * 2);
    outbound = new Hmat<unsigned short>(width, width);

}

int TF_HalfPredict::predict()
{
    int i,j;
    // gerenate input
    // ================================================================================
    TF_Operation *input_op = TF_GraphOperationByName(graph, "Placeholder");
    // printf("input_op has %i inputs\n", TF_OperationNumOutputs(input_op));
    float* raw_input_data = (float*)malloc(width*2*width*2 * sizeof(float));
    for(i = 0; i < width*2; i++)
    {
        for(j = 0; j < width*2; j++)
        {
            raw_input_data[j+i*width*2] = (*inbound)[i][j]/mm.norm_value;
        }
    }
    int64_t* raw_input_dims = (int64_t*)malloc(4 * sizeof(int64_t));
    raw_input_dims[0] = 1;
    raw_input_dims[1] = width*2;
    raw_input_dims[2] = width*2;
    raw_input_dims[3] = 1;

    /*
    TF_CAPI_EXPORT extern TF_Tensor* TF_NewTensor(
      TF_DataType,
      const int64_t* dims, int num_dims,
      void* data, size_t len,
      void (*deallocator)(void* data, size_t len, void* arg),
      void* deallocator_arg);
    */
    // prepare inputs
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT,
                                         raw_input_dims, 4,
                                         raw_input_data, width*2*width*2 * sizeof(float),
                                         deallocator,
                                         NULL
                                        );


    // void* input_data = TF_TensorData(input_tensor);
    // printf("input_data[0] = %f\n", ((float*)input_data)[0]);
    // printf("input_data[1] = %f\n", ((float*)input_data)[1]);


    TF_Output* run_inputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
    run_inputs[0].oper = input_op;
    run_inputs[0].index = 0;

    TF_Tensor** run_inputs_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
    run_inputs_tensors[0] = input_tensor;

    // prepare outputs
    // ================================================================================
     
    TF_Operation* output_op;
    if (width >= 16)
    {
        output_op = TF_GraphOperationByName(graph, "main_full/conv11/BiasAdd");
        // output_op = TF_GraphOperationByName(graph, "main_full/postprocess_conv11/BiasAdd");
    }
    else
    {
        output_op = TF_GraphOperationByName(graph, "main_full/conv11/BiasAdd");
    }
    // printf("output_op has %i outputs\n", TF_OperationNumOutputs(output_op));

    TF_Output* run_outputs = (TF_Output*)malloc(1 * sizeof(TF_Output));
    run_outputs[0].oper = output_op;
    run_outputs[0].index = 0;


    TF_Tensor** run_output_tensors = (TF_Tensor**)malloc(1 * sizeof(TF_Tensor*));
    float* raw_output_data = (float*)malloc(width*width * sizeof(float));
    raw_output_data[0] = 1.f;
    int64_t* raw_output_dims = (int64_t*)malloc(4 * sizeof(int64_t));
    raw_output_dims[0] = 1;
    raw_output_dims[1] = width;
    raw_output_dims[2] = width;
    raw_output_dims[3] = 1;

    TF_Tensor* output_tensor = TF_NewTensor(TF_FLOAT,
                                          raw_output_dims, 4,
                                          raw_output_data, width*width * sizeof(float),
                                          deallocator,
                                          NULL
                                         );
    run_output_tensors[0] = output_tensor;

    // run network
    // ================================================================================
    TF_SessionRun(sess,
                /* RunOptions */         NULL,
                /* Input tensors */      run_inputs, run_inputs_tensors, 1,
                /* Output tensors */     run_outputs, run_output_tensors, 1,
                /* Target operations */  NULL, 0,
                /* RunMetadata */        NULL,
                /* Output status */      status);
    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "ERROR: Unable to run output_op: %s\n", TF_Message(status));
        return 1;
    }

    // printf("output-tensor has %i dims\n", TF_NumDims(run_output_tensors[0]));

    void* output_data = TF_TensorData(run_output_tensors[0]);
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            (*outbound)[i][j] = (*outbound)[i][j] = (unsigned short)round(((float*)output_data)[j+i*width] * mm.norm_value);
        }
    }

    // free((void*) raw_input_data);
    // free((void*) raw_output_data);
    // free((void*) raw_input_dims);
    // free((void*) run_inputs);

    free((void*) raw_input_dims);
    free((void*) raw_output_dims);
    free((void*) run_inputs);
    free((void*) run_outputs);
    free((void*) run_inputs_tensors);
    free((void*) run_output_tensors);
}
