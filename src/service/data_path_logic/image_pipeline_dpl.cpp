#include <cascade/service.hpp>
#include <cascade/data_path_logic_interface.hpp>
#include <iostream>
#include <cascade/config.h>
#include <mxnet-cpp/MxNetCpp.h>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <string>
#include "cnn_classifier_dpl.hpp"
#include "scheduler_dpl.hpp"

namespace derecho{
namespace cascade{

// #define MY_PREFIX   "/image_pipeline"
#define MY_PREFIX "/Ithaca/farm1"
#define MY_UUID     "48e60f7c"
#define MY_DESC     "DLL DPL that performs inference on image frames collected on the farm"

#define DFG_CONFIG  "dfg.json"
#define EVALUATION 1 

DFGDescriptor dfg_descriptor; 
template <typename... CascadeTypes>
std::tuple<uint32_t, uint32_t> select_shard(CascadeContext<CascadeTypes...>& ctxt, std::string object_pool_id, std::string key);
// Helper function for changing prefix
int nthOccurrence(const std::string& str, const std::string& findMe, int nth);

std::unordered_set<std::string> list_prefixes() {
    return {MY_PREFIX};
}

std::string get_uuid() {
    return MY_UUID;
}

std::string get_description() {
    return MY_DESC;
}


void initialize(ICascadeContext* ctxt) {
    std::ifstream input_conf(DFG_CONFIG);
    json dfg_conf = json::parse(input_conf); 
    dfg_descriptor = DFGDescriptor(dfg_conf);
    // create object pool correspond to the dfg tasks
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeMetadataWithStringKey,VolatileCascadeStoreWithStringKey,
                                                    PersistentCascadeStoreWithStringKey>*>(ctxt);
    typed_ctxt->store_dfg(dfg_descriptor);
    dbg_default_trace("\n\n [CascadeContext] initilization");

    dfg_descriptor.dump();
}

enum TypeFlag {
    kFloat32 = 0,
    kFloat64 = 1,
    kFloat16 = 2,
    kUint8 = 3,
    kInt32 = 4,
    kInt8 = 5,
    kInt64 = 6,
};

#define DPL_CONF_FLOWER_SYNSET  "CASCADE/flower_synset"
#define DPL_CONF_FLOWER_SYMBOL  "CASCADE/flower_symbol"
#define DPL_CONF_FLOWER_PARAMS  "CASCADE/flower_params"
#define DPL_CONF_PET_SYNSET  "CASCADE/pet_synset"
#define DPL_CONF_PET_SYMBOL  "CASCADE/pet_symbol"
#define DPL_CONF_PET_PARAMS  "CASCADE/pet_params"
#define DPL_CONF_USE_GPU        "CASCADE/use_gpu"


class InferenceEngine {
private:
    /**
     * the synset explains inference result.
     */
    std::vector<std::string> synset_vector;
    /**
     * symbol
     */
    mxnet::cpp::Symbol net;
    /**
     * argument parameters
     */
    std::map<std::string, mxnet::cpp::NDArray> args_map;
    /**
     * auxliary parameters
     */
    std::map<std::string, mxnet::cpp::NDArray> aux_map;
    /**
     * global ctx
     */
    const mxnet::cpp::Context& global_ctx;
    /**
     * the input shape
     */
    mxnet::cpp::Shape input_shape;
    /**
     * argument arrays
     */
    std::vector<mxnet::cpp::NDArray> arg_arrays;
    /**
     * gradient arrays
     */
    std::vector<mxnet::cpp::NDArray> grad_arrays;
    /**
     * ??
     */
    std::vector<mxnet::cpp::OpReqType> grad_reqs;
    /**
     * auxliary array
     */
    std::vector<mxnet::cpp::NDArray> aux_arrays;
    /**
     * client data
     */
    mxnet::cpp::NDArray client_data;
    /**
     * the work horse: mxnet executor
     */
    std::unique_ptr<mxnet::cpp::Executor> executor_pointer;    
    
    void load_synset(const std::string& synset_file) {
        // dbg_default_trace("synset file="+synset_file);
        std::ifstream fin(synset_file);
        synset_vector.clear();
        for(std::string syn;std::getline(fin,syn);) {
            synset_vector.push_back(syn);
        }
        fin.close();
    }

    void load_symbol(const std::string& symbol_file) {
        // dbg_default_trace("symbol file="+symbol_file);
        this->net = mxnet::cpp::Symbol::Load(symbol_file);
    }

    void load_params(const std::string& params_file) {
        // dbg_default_trace("params file="+params_file);
        auto parameters = mxnet::cpp::NDArray::LoadToMap(params_file);
        for (const auto& kv : parameters) {
            if (kv.first.substr(0, 4) == "aux:") {
                auto name = kv.first.substr(4, kv.first.size() - 4);
                this->aux_map[name] = kv.second.Copy(global_ctx);
            } else if (kv.first.substr(0, 4) == "arg:") {
                auto name = kv.first.substr(4, kv.first.size() - 4);
                this->args_map[name] = kv.second.Copy(global_ctx);
            }
        }
        mxnet::cpp::NDArray::WaitAll();
        this->args_map["data"] = mxnet::cpp::NDArray(input_shape, global_ctx, false, kFloat32);
        mxnet::cpp::Shape label_shape(input_shape[0]);
        this->args_map["softmax_label"] = mxnet::cpp::NDArray(label_shape, global_ctx, false);
        this->client_data = mxnet::cpp::NDArray(input_shape, global_ctx, false, kFloat32);
    }

    bool load_model(const std::string& synset_file,
                    const std::string& symbol_file,
                    const std::string& params_file) {
        try {
            load_synset(synset_file);
            load_symbol(symbol_file);
            load_params(params_file);
            mxnet::cpp::NDArray::WaitAll();

            dbg_default_trace("creating executor.");
            this->net.InferExecutorArrays(
                    global_ctx, &arg_arrays, &grad_arrays, &grad_reqs, &aux_arrays,
                    args_map, std::map<std::string, mxnet::cpp::NDArray>(),
                    std::map<std::string, mxnet::cpp::OpReqType>(), aux_map);
            for(auto& i : grad_reqs)
                i = mxnet::cpp::OpReqType::kNullOp;
            this->executor_pointer.reset(new mxnet::cpp::Executor(
                    net, global_ctx, arg_arrays, grad_arrays, grad_reqs, aux_arrays));
            // dbg_default_trace("load_model() finished.");
            return true;
        } catch(const std::exception& e) {
            std::cerr << "Load model failed with exception " << e.what() << std::endl;
            return false;
        } catch(...) {
            std::cerr << "Load model failed with unknown exception." << std::endl;
            return false;
        }
    }

public:
    InferenceEngine(const mxnet::cpp::Context& ctxt,
                    const std::string& synset_file,
                    const std::string& symbol_file,
                    const std::string& params_file):
        global_ctx(ctxt),
        input_shape(std::vector<mxnet::cpp::index_t>({1, 3, 224, 224})) {
        // dbg_default_trace("loading model begin.");
        load_model(synset_file, symbol_file, params_file);
        // dbg_default_trace("loading model end.");
    }

    // do the inference.
    std::pair<std::string,double> infer(const ImageFrame& frame) {
#ifdef EVALUATION
        uint64_t start_ns = get_time();
#endif
        // copy to input layer:                                     
        FrameData *fd = reinterpret_cast<FrameData*>(frame.bytes);
        args_map["data"].SyncCopyFromCPU(reinterpret_cast<const mx_float*>(fd->data), input_shape.Size());
        this->executor_pointer->Forward(false);
        mxnet::cpp::NDArray::WaitAll();
        // extract the result
        auto output_shape = executor_pointer->outputs[0].GetShape();
        mxnet::cpp::NDArray output_in_cpu(output_shape,mxnet::cpp::Context::cpu());
        executor_pointer->outputs[0].CopyTo(&output_in_cpu);
        mxnet::cpp::NDArray::WaitAll();
        mx_float max = -1e10;
        int idx = -1;
        for(unsigned int jj = 0; jj < output_shape[1]; jj++) {
            if(max < output_in_cpu.At(0, jj)) {
                max = output_in_cpu.At(0, jj);
                idx = static_cast<int>(jj);
            }
        }
#ifdef EVALUATION
        uint64_t end_ns = get_time();
#endif
        return {synset_vector[idx],max};
    }
};

class ClassifierTrigger: public OffCriticalDataPathObserver {
private:
    // InferenceEngine flower_ie;
    // InferenceEngine pet_ie;
    mutable std::mutex p2p_send_mutex;
#ifdef EVALUATION
    int sock_fd;
    struct sockaddr_in serveraddr;
#endif
public:
    ClassifierTrigger (): 
        OffCriticalDataPathObserver()// ,
        // flower_ie(derecho::getConfString(DPL_CONF_FLOWER_SYNSET),derecho::getConfString(DPL_CONF_FLOWER_SYMBOL),derecho::getConfString(DPL_CONF_FLOWER_PARAMS)),
        // pet_ie(derecho::getConfString(DPL_CONF_PET_SYNSET),derecho::getConfString(DPL_CONF_PET_SYMBOL),derecho::getConfString(DPL_CONF_PET_PARAMS))
        {
#ifdef EVALUATION
#define DPL_CONF_REPORT_TO  "CASCADE/report_to"
            uint16_t port;
            struct hostent *server;
            std::string hostname;
            std::string report_to = derecho::getConfString(DPL_CONF_REPORT_TO);
            hostname = report_to.substr(0,report_to.find(":"));
            port = (uint16_t)std::stoi(report_to.substr(report_to.find(":")+1));
            sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
            if (sock_fd < 0) {
                std::cerr << "Faile to open socket" << std::endl;
                return;
            }
            server = gethostbyname(hostname.c_str());
            if (server == nullptr) {
                std::cerr << "Failed to get host:" << hostname << std::endl;
            }

            bzero((char *) &serveraddr, sizeof(serveraddr));
            serveraddr.sin_family = AF_INET;
            bcopy((char *)server->h_addr, 
              (char *)&serveraddr.sin_addr.s_addr, server->h_length);
            serveraddr.sin_port = htons(port);
#endif
    }

    virtual void operator () (const std::string& key_string,
                              persistent::version_t version,
                              const mutils::ByteRepresentable* const value_ptr,
                              ICascadeContext* ctxt,
                              uint32_t worker_id) override {
        auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeMetadataWithStringKey,
                                            VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey>*>(ctxt);
        /* step 1 prepare context */
        bool use_gpu = derecho::hasCustomizedConfKey(DPL_CONF_USE_GPU)?derecho::getConfBoolean(DPL_CONF_USE_GPU):false;
        if (use_gpu && typed_ctxt->resource_descriptor.gpus.size()==0) {
            dbg_default_error("Worker{}: GPU is requested but no GPU found...giving up on processing data.",worker_id);
            return;
        }
        static thread_local const mxnet::cpp::Context mxnet_ctxt(
            use_gpu? mxnet::cpp::DeviceType::kGPU : mxnet::cpp::DeviceType::kCPU,
            use_gpu? typed_ctxt->resource_descriptor.gpus[worker_id % typed_ctxt->resource_descriptor.gpus.size()]:0);

        /* create inference engines */
        static thread_local InferenceEngine flower_ie(
                mxnet_ctxt,
                derecho::getConfString(DPL_CONF_FLOWER_SYNSET),
                derecho::getConfString(DPL_CONF_FLOWER_SYMBOL),
                derecho::getConfString(DPL_CONF_FLOWER_PARAMS));
        
        static thread_local InferenceEngine pet_ie(
                mxnet_ctxt,
                derecho::getConfString(DPL_CONF_PET_SYNSET),
                derecho::getConfString(DPL_CONF_PET_SYMBOL),
                derecho::getConfString(DPL_CONF_PET_PARAMS));
        
        VolatileCascadeStoreWithStringKey::ObjectType *vcss_value = const_cast<VolatileCascadeStoreWithStringKey::ObjectType*>(
                                                                    reinterpret_cast<const VolatileCascadeStoreWithStringKey::ObjectType *>(value_ptr));
        std::unique_ptr<ImageFrame> frame = std::make_unique<ImageFrame>(vcss_value->get_key_ref(),vcss_value->blob);
        
        std::string name;
        double soft_max;
        int dfg_pos = nthOccurrence(frame->key, "/", 1);
        std::string dfg_name = frame->key.substr(0, dfg_pos);
        size_t node_pos = frame->key.rfind('/');
        std::string current_node_name = frame->key.substr(dfg_pos, node_pos-dfg_pos);
        
#ifdef EVALUATION
        uint64_t before_inference_ns = get_time();
#endif
        if (current_node_name == "/pet") {
            std::tie(name,soft_max) = pet_ie.infer(*frame);
        } else {
            std::tie(name,soft_max) = flower_ie.infer(*frame);
        }
#ifdef EVALUATION
        uint64_t after_inference_ns = get_time();
#endif
        std::cout << "\033[1;31m"<<"\n\n FINISHED INFERENCE for: "<< key_string << "\033[0m" << std::endl;

        // First ceck if there is corresponding dfg  defined previously
        std::unordered_map<std::string,DFGDescriptor>::const_iterator got = typed_ctxt->get_cached_dfgs().find (dfg_name);
        if (got == typed_ctxt->get_cached_dfgs().end() ){
            // Case1. there is no corresponding dfg, then task end here.
            std::string reput_key = frame->key+"/infer";
            VolatileCascadeStoreWithStringKey::ObjectType obj(reput_key,name.c_str(),name.size());
            std::lock_guard<std::mutex> lock(p2p_send_mutex);
#ifdef EVALUATION
            CloseLoopReport clr;
            FrameData* fd = reinterpret_cast<FrameData*>(frame->bytes);
            clr.photo_id = fd->photo_id;
            clr.inference_us = (after_inference_ns-before_inference_ns)/1000;
#endif
            auto result = typed_ctxt->get_service_client_ref().template put<VolatileCascadeStoreWithStringKey>(obj,0,0,false);
            for (auto& reply_future:result.get()) {
                auto reply = reply_future.second.get();
                dbg_default_debug("node({}) replied with version:({:x},{}us)",reply_future.first,std::get<0>(reply),std::get<1>(reply));
            }

#ifdef EVALUATION
            uint64_t after_put_ns = get_time();
            clr.put_us = (after_put_ns-after_inference_ns)/1000;
            int serverlen = sizeof(serveraddr);
            size_t ns = sendto(sock_fd,(void*)&clr,sizeof(clr),0,(const sockaddr*)&serveraddr,serverlen);
            if (ns < 0) {
                std::cerr << "Failed to report error" << std::endl;
            }
#endif
        }else{
            // Case2. If there is corresponding dfg, then use it and RUN SCHEDULER
            std::cout << "\033[1;31m"<<"\n\n Found the dfg: " << dfg_name << ", current node name is : "<< current_node_name << "\033[0m" << std::endl;

            for(auto& dfg_it : typed_ctxt->get_cached_dfgs()){
                DFGDescriptor dfg = static_cast<DFGDescriptor>(dfg_it.second);
                std::string obj_key = frame->key.substr( node_pos );
                std::vector<std::string> output_objectpools = dfg.get_output_objectpools(current_node_name);
                for (const auto& e : output_objectpools) {
                    std::string object_pool_id = static_cast<std::string> (e);
                    std::string reput_key = object_pool_id + obj_key;
                    // PersistentCascadeStoreWithStringKey::ObjectType obj(reput_key,name.c_str(),name.size());
                    VolatileCascadeStoreWithStringKey::ObjectType obj(reput_key,name.c_str(),name.size());
                    std::lock_guard<std::mutex> lock(p2p_send_mutex);
#ifdef EVALUATION
                    CloseLoopReport clr;
                    FrameData* fd = reinterpret_cast<FrameData*>(frame->bytes);
                    clr.photo_id = fd->photo_id;
                    clr.inference_us = (after_inference_ns-before_inference_ns)/1000;
#endif
                    std::tuple<uint32_t, uint32_t> scheduled_loc = select_shard<VolatileCascadeMetadataWithStringKey,VolatileCascadeStoreWithStringKey,
                                                                    PersistentCascadeStoreWithStringKey>(*typed_ctxt,object_pool_id,reput_key);
                    uint32_t subgroup_index = std::get<0>(scheduled_loc);
                    uint32_t shard_index = std::get<1>(scheduled_loc);
                    
                    auto result = typed_ctxt->get_service_client_ref().template put<VolatileCascadeStoreWithStringKey>(obj,subgroup_index,shard_index,false);
                    for (auto& reply_future:result.get()) {
                        auto reply = reply_future.second.get();
                        dbg_default_debug("node({}) replied with version:({:x},{}us)",reply_future.first,std::get<0>(reply),std::get<1>(reply));
                    }
                
#ifdef EVALUATION
                    uint64_t after_put_ns = get_time();
                    clr.put_us = (after_put_ns-after_inference_ns)/1000;
                    int serverlen = sizeof(serveraddr);
                    size_t ns = sendto(sock_fd,(void*)&clr,sizeof(clr),0,(const sockaddr*)&serveraddr,serverlen);
                    if (ns < 0) {
                        std::cerr << "Failed to report error" << std::endl;
                    }
#endif
                }
            }
        }

    }

    virtual ~ClassifierTrigger() {
#ifdef EVALUATION
        close(sock_fd);
#endif
    }
};

// TODO: question: register_triggers could be done at CascadeContext initialization based on dfg???
void register_triggers(ICascadeContext* ctxt) {
    // Please make sure the CascadeContext type matches the CascadeService type, which is defined in server.cpp if you
    // use the default cascade service binary.
    dbg_default_trace("\n\n [CascadeContext] registered trigger");
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeMetadataWithStringKey,VolatileCascadeStoreWithStringKey,
                                                    PersistentCascadeStoreWithStringKey>*>(ctxt);
    // typed_ctxt->register_prefixes({MY_PREFIX},MY_UUID,std::make_shared<ClassifierTrigger>());
    for (auto const& dfg_descriptor : typed_ctxt->get_cached_dfgs())
    {
        DFGDescriptor dfg = static_cast<DFGDescriptor> (dfg_descriptor.second);
        typed_ctxt->register_dfg( dfg , std::make_shared<ClassifierTrigger>());
    }
}

void unregister_triggers(ICascadeContext* ctxt) {
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeMetadataWithStringKey,VolatileCascadeStoreWithStringKey,
                                                    PersistentCascadeStoreWithStringKey>*>(ctxt);
    typed_ctxt->unregister_prefixes({MY_PREFIX},MY_UUID);
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho

