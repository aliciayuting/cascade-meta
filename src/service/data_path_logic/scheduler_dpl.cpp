#include <cascade/config.h>
#include <cascade/service_server_api.hpp>
#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

/**
 * This is an prototype for the scheduler dll
 */
namespace derecho{
namespace cascade{
void on_cascade_initialization() {
    std::cout << "[Scheduler dll]: initialize the scheduler here." << std::endl;
}

void on_cascade_exit() {
    std::cout << "[Scheduler dll]: destroy the scheduler before exit." << std::endl;
}

#define DPL_CONF_USE_GPU        "CASCADE/use_gpu"

#define AT_UNKNOWN      (0)
#define AT_PET_BREED    (1)
#define AT_FLOWER_NAME  (2)
#define AT_SCHEDULER    (3)


class Scheduler: public OffCriticalDataPathObserver {
// private:
//     mutable std::mutex p2p_send_mutex;
public:
    Scheduler (): 
        OffCriticalDataPathObserver()
        {}

    // primative relay the action 
    virtual void operator () (Action&& action, ICascadeContext* cascade_ctxt, uint32_t worker_id) {
        auto* ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey,VolatileCascadeMetadataWithStringKey>*>(cascade_ctxt);
        std::cout << "[Scheduler] I started to scheduling: " << action.action_type << "; Relay the execution to a worker" <<  std::endl;
        // Run algorithm to find the subgroup's shard to execute

        ctxt->post(std::move(action));
    }

    // virtual ~Scheduler() {
    // }
};

std::shared_ptr<OffCriticalDataPathObserver> get_off_critical_data_path_observer() {
    return std::make_shared<Scheduler>();
}

} // namespace cascade
} // namespace derecho



/***
 *  dplm branch
#include <cascade/data_path_logic_interface.hpp>
#include <iostream>

namespace derecho{
namespace cascade{

#define MY_PREFIX   "/scheduler"
#define MY_UUID     "48e60f7c-8500-11eb-8755-0242ac110002"
#define MY_DESC     "Prototype of Scheduler currently only relay and trigger next task after prefix " MY_PREFIX " on console."

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
    // nothing to initialize
    return;
}

class Scheduler: public OffCriticalDataPathObserver {
    virtual void operator () (const std::string& key_string,
                              persistent::version_t version,
                              const mutils::ByteRepresentable* const value_ptr,
                              ICascadeContext* ctxt,
                              uint32_t worker_id) override {
        std::cout << "[Running Scheduler]: I(" << worker_id << ") received an object with key=" << key_string << std::endl;
    }
};

void register_triggers(ICascadeContext* ctxt) {
    // Please make sure the CascadeContext type matches the CascadeService type, which is defined in server.cpp if you
    // use the default cascade service binary.
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey,>*>(ctxt);
    typed_ctxt->register_prefixes({MY_PREFIX},MY_UUID,std::make_shared<ConsolePrinterOCDPO>());
}

void unregister_triggers(ICascadeContext* ctxt) {
    auto* typed_ctxt = dynamic_cast<CascadeContext<VolatileCascadeStoreWithStringKey,PersistentCascadeStoreWithStringKey>*>(ctxt);
    typed_ctxt->unregister_prefixes({MY_PREFIX},MY_UUID);
}

void release(ICascadeContext* ctxt) {
    // nothing to release
    return;
}

} // namespace cascade
} // namespace derecho
***/