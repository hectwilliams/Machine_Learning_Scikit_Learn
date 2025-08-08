'''
  5.

    Why would you go through the hassle of converting all your data to the Example protobuf format?

    Why not use your own protobuf definition?

    - Example protobuf is the most common used protobuf in TFRecord files. It accepts Int64, BytesList and FloatList which are the base unit type for mostly all data instances

    - Creating a protobuf definition file requires new classes to be compiled. Tensorflow provides Example and SequenceExample profotbuf class which should solve 99.9 percent of problems. 

'''