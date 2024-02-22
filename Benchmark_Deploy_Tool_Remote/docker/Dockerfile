FROM node:16.18.1-alpine3.17
RUN npm config set registry https://registry.npm.taobao.org/
RUN mkdir -p /hyperledger/caliper/workspace && npm -g config set user root && npm i -g @hyperledger/caliper-cli@0.5.0 node-gyp --registry https://registry.npm.taobao.org/
# COPY node_modules /hyperledger/caliper/workspace
# COPY package-lock.json /hyperledger/caliper/workspace
# COPY package.json /hyperledger/caliper/workspace
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apk/repositories
RUN apk add python3 gcc g++ make libffi-dev openssl-dev libtool
RUN caliper bind --caliper-bind-sut fabric:2.4 --caliper-bind-args=-g