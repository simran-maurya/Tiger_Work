
FROM continuumio/miniconda3

ADD  dist/files_for_housing-0.1.tar.gz home/

WORKDIR /home/files_for_housing-0.1


ADD ./test ./test
ADD ./enviornment_file ./enviornment_file
ADD ./docs ./docs
ADD entrypoint.sh .

RUN conda env create -f enviornment_file/env.yml

RUN conda run -n mle-dev pip install .
RUN chmod +x entrypoint.sh
CMD /usr/bin/tail -f /dev/null
ENTRYPOINT [ "./entrypoint.sh" ]
