"""
    Q.
        When should you use gRPC API rather than the REST API to query a model served by TF Serving?

    A.
        REST API is not efficient (REST is text-based JSON formatted data) requiring serialization/deserialization ( floats (raw)-> strings (JSON) -> float(raw)).
        The gRPC API expects serialized protocol buffer which can transfer large numpy arrays far better off than the latency rich REST API.

    

"""