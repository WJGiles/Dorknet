FROM intelpython/intelpython3_core

RUN apt update
RUN apt -y install nano git
RUN pip install opencv-python pillow tqdm line-profiler cython h5py

RUN git clone https://github.com/pydata/numexpr
RUN cd numexpr && python setup.py build && python setup.py install

WORKDIR /Dorknet

CMD cd layers && python setup.py build_ext install && cd /Dorknet && /bin/bash
