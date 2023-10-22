<template>
    <div id="graph-root">
      <div id="graph-input">
        <el-input
          type="textarea"
          :rows="3"
          placeholder="Please input the question"
          v-model="textarea"
          resize="none"
          @keydown.enter.native="sendMessage($event)"
        >
        </el-input>
      </div>
      <div id="graph-content">
        <RelationGraph
          ref="seeksRelationGraph"
          :options="graphOptions"
          :on-node-click="onNodeClick"
          :on-line-click="onLineClick"
          style="font-size: 20px"
        />
      </div>
    </div>
  </template>
  
  <script>
  import RelationGraph from "relation-graph";
  
  export default {
    name: "graph",
    components: { RelationGraph },
    data() {
      return {
        textarea: "",
        graphOptions: {
          allowSwitchLineShape: true,
          allowSwitchJunctionPoint: true,
          defaultJunctionPoint: "border",
          layouts: [
            {
              layoutName: "force",
            },
          ],
        },
        nodes: [],
        links: [],
      };
    },
    mounted() {
      let __graph_json_data = {
        rootId: "空节点",
        nodes: [
          {
            id: "空节点",
            text: "空节点",
            color: "#1167be",
            borderColor: "#1167be",
            width: "80",
            height: "80",
          },
        ],
        links: [],
        layouts: { layoutName: "force" },
      };
      this.showSeeksGraph(__graph_json_data);
    },
    methods: {
      showSeeksGraph(__graph_json_data) {
        this.$refs.seeksRelationGraph.setJsonData(
          __graph_json_data,
          (seeksRGGraph) => {
            // Called when the relation-graph is completed
          }
        );
      },
      onNodeClick(nodeObject, $event) {
        console.log("onNodeClick:", nodeObject);
      },
      onLineClick(lineObject, $event) {
        console.log("onLineClick:", lineObject);
      },
      sendMessage(event) {
        // 阻止默认事件
        event.preventDefault();
  
        let that = this;
  
        let question = this.textarea;
        // this.textarea = "";
  
        // 将问题发给后端并返回答案
        this.$axios
          .post("http://127.0.0.1:5000/getGraph", {
            question: question,
          })
          .then((res) => {
            let answer = res.data;
            if (answer == "NotFound") {
              that.$message({
                showClose: true,
                message: "Answer Not Found!",
                type: "warning",
                duration:1000,
              });
              console.log(answer);
            } else {
              let __graph_json_data = {
                rootId: "",
                nodes: [],
                links: [],
                layouts: { layoutName: "force" },
              };
  
              answer.forEach((elem) => {
                let node = {
                  id: elem.entity,
                  text: elem.entity,
                  color: "#1167be",
                  borderColor: "#1167be",
                  width: "50",
                  height: "50",
                  innerHTML:
                    '<div style="overflow: hidden;white-space: nowrap;text-overflow: ellipsis;line-height:50px;font-size:12px;border-radius:50%">' +
                    elem.entity +
                    "</div>",
                };
                __graph_json_data.rootId = elem.entity;
                __graph_json_data.nodes.push(node);
  
                elem.object.forEach((oElem) => {
                  if (oElem instanceof Array) {
                    oElem.forEach((objElem) => {
                      let obNode = {
                        id: objElem,
                        color: "#1167be",
                        borderColor: "#1167be",
                        width: "50",
                        height: "50",
                        innerHTML:
                          '<div style="overflow: hidden;white-space: nowrap;text-overflow: ellipsis;line-height:50px;font-size:12px;border-radius:50%">' +
                          objElem +
                          "</div>",
                      };
  
                      let link = {
                        from: elem.entity,
                        to: objElem,
                        text: elem.relation,
                        color: "#43a2f1",
                      };
  
                      __graph_json_data.nodes.push(obNode);
                      __graph_json_data.links.push(link);
                    });
                  } else {
                    let obNode = {
                      id: oElem,
                      width: "50",
                      height: "50",
                      color: "#1167be",
                      borderColor: "#1167be",
                      innerHTML:
                        '<div style="overflow: hidden;white-space: nowrap;text-overflow: ellipsis;line-height:50px;font-size:12px;border-radius:50%">' +
                        oElem +
                        "</div>",
                    };
  
                    let link = {
                      from: elem.entity,
                      to: oElem,
                      text: elem.relation,
                      color: "#43a2f1",
                    };
  
                    __graph_json_data.nodes.push(obNode);
                    __graph_json_data.links.push(link);
                  }
                });
  
                this.showSeeksGraph(__graph_json_data);
              });
            }
          });
      },
    },
  };
  </script>
  
  <style scoped lang="less">
  #graph-root {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    #graph-content {
      width: 100%;
      height: 90%;
      #node {
        font-size: 14px;
        color: brown !important;
        background: #1167be;
        width: 50px;
        height: 50px;
        overflow: hidden;
      }
    }
    #graph-input {
      width: 100%;
      height: 10%;
      min-height: 100px;
      box-sizing: border-box;
      padding: 10px;
      border-bottom: 1px solid #c7c7c7;
    }
  }
  </style>
  