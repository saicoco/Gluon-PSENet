CXXFLAGS = -I include -std=c++14 -O3 -Wl,-undefined,dynamic_lookup

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp
OPENCV = `pkg-config --cflags --libs opencv`

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)