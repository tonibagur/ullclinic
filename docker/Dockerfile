FROM nvidia/cuda:8.0-cudnn6-devel

RUN apt-get update
RUN apt-get install -y git python-pip python-dev openjdk-8-jdk curl
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN mkdir /src
RUN cd /src && curl https://github.com/bazelbuild/bazel/releases/download/0.5.4/bazel-0.5.4-installer-linux-x86_64.sh > bazel-0.5.4-installer-linux-x86_64.sh
#RUN /tmp/bazel-0.5.4-installer-linux-x86_64.sh
#RUN apt-get update && apt-get install -y bazel


