<template>
  <div id="chat-root">
    <div id="chat-left">
      <div id="chat-content" ref="scroll">
        <!-- 系统提示
      <div id="chat-tips">
        Intelligent assistant enters the conversation to serve you
      </div> -->
        <!-- 滚动标签 -->
        <div v-for="(itemc, indexc) in recordContent" :key="indexc">
          <!-- 对方 -->
          <div
            class="word animate__animated animate__fadeInLeft"
            v-if="itemc.id == 2"
            :class="indexc == 0 ? 'info-first' : 'info-others'"
          >
            <img :src="require(`@/assets/${itemc.headUrl}`)" />
            <div class="info">
              <p class="nickname">
                {{ itemc.nickName }}
              </p>
              <div
                class="info-content animate__animated animate__fadeInLeft"
                :class="
                  indexc == 0 ? 'info-content-first' : 'info-content-others'
                "
              >
                <span v-if="itemc.origin == 0 && itemc.contactText != 'None'">{{
                  itemc.contactText
                }}</span>
                <span v-if="itemc.origin == 0 && itemc.contactText == 'None'"
                  ><div class="wait">
                    <div></div>
                    <div></div>
                    <div></div>
                    <div></div>
                    <div></div></div
                ></span>
                <ChatKGText
                  :contactText="itemc.contactText"
                  v-if="itemc.origin == 1"
                  @isShowKG="isShowKG"
                ></ChatKGText>
                <ChatMap
                  :contactText="itemc.contactText"
                  v-if="itemc.origin == 2"
                  @isShowMap="isShowMap"
                ></ChatMap>
                <ChatRuleBox
                  :contactText="itemc.contactText"
                  v-if="itemc.origin == 3"
                  @isShowRuleTips="isShowRuleTips"
                ></ChatRuleBox>
              </div>
              <div
                class="message_time animate__animated animate__fadeInLeft"
                :class="
                  indexc == 0 ? 'info-content-first' : 'info-content-others'
                "
              >
                {{ itemc.time }}
              </div>
            </div>
          </div>

          <!-- 我的 -->
          <div
            class="word-my animate__animated animate__fadeInRight info-others"
            v-else
          >
            <div class="info">
              <p class="nickname">
                {{ itemc.nickName }}
              </p>
              <div
                class="
                  info-content
                  animate__animated animate__fadeInRight
                  info-content-others
                "
              >
                {{ itemc.contactText }}
              </div>
              <div
                class="
                  Sender_time
                  animate__animated animate__fadeInRight
                  info-content-others
                "
              >
                {{ itemc.time }}
              </div>
            </div>
            <img :src="require(`@/assets/${itemc.headUrl}`)" />
          </div>
        </div>
      </div>

      <div id="chat-scroll">
        <div class="chat-scroll-button" @click="left_scroll">
          <img :src="require('@/assets/left_arrow.png')" />
        </div>
        <ul
          class="animate__animated animate__fadeIn"
          id="scroll-tips"
          ref="scroll_elem"
        >
          <li v-for="(item, index) in tipsList" :key="index">
            <el-button round @click="sendTips(item, $event)">{{
              item
            }}</el-button>
          </li>
        </ul>
        <div class="chat-scroll-button" @click="right_scroll">
          <img :src="require('@/assets/right_arrow.png')" />
        </div>
      </div>

      <div id="chat-input">
        <div class="chat-input-box">
          <el-input
            type="textarea"
            :rows="4"
            placeholder="Please input the question"
            v-model="textarea"
            resize="none"
            @keydown.enter.native="sendMessage"
          >
          </el-input>
        </div>
        <div class="chat-input-button">
          <el-button
            icon="el-icon-position"
            circle
            @click="sendMessage"
          ></el-button>
        </div>
      </div>
    </div>
    <div id="chat-right">
      <div id="chat-card">
        <div id="chat-item-list" class="chat-item-list-top">
          <el-card class="box-card">
            <div slot="header" class="clearfix">
              <span>FAQ</span>
            </div>
            <div class="content_scroll">
              <div
                v-for="(item, index) in faqList"
                :key="index"
                class="text item"
                :title="item"
                @click="faqSend(item)"
              >
                {{ item }}
              </div>
            </div>
          </el-card>
        </div>
      </div>
      <div id="chat-card" class="chat-contact">
        <div id="chat-item-list" class="chat-item-list-bottom">
          <el-card class="box-card">
            <div slot="header" class="clearfix">
              <span>CONTACT POLYU</span>
            </div>
            <div class="contact item">PHONE: (852) 2766 5111</div>
            <div class="contact item">EMAIL: web.master@polyu.edu.hk</div>
          </el-card>
        </div>
      </div>
    </div>
    <div id="chat-entities" v-if="showRuleTip">
      <div id="chat-entities-box">
        <div id="chat-entities-box-search">
          <el-input
            v-model="entity_seach_input"
            placeholder="Search entities"
          ></el-input>
        </div>
        <div id="chat-entities-box-list">
          <div>
            <ul>
              <li v-for="(item, index) in filterEntities" :key="index" @click="sendEntity(item)">
                {{ item }}
              </li>
            </ul>
          </div>
        </div>
        <div id="chat-entities-box-close" @click="closeRuleTip">CLOSE</div>
      </div>
    </div>
  </div>
</template>

<script>
import ChatKGText from "../components/ChatKGText.vue";
import ChatMap from "../components/ChatMap.vue";
import ChatRuleBox from "../components/ChatRuleBox.vue";
export default {
  name: "ChatBox",
  data() {
    return {
      activeIndex: "1",
      recordContent: [],
      textarea: "",
      currentTime: "",
      tipsList: [],
      faqList: [],
      conId: "",
      ruleContent: [],
      showRuleTip: false,
      entity_seach_input: "",
    };
  },
  components: {
    ChatKGText,
    ChatMap,
    ChatRuleBox,
  },
  methods: {
    handleSelect(key, keyPath) {
      console.log(key, keyPath);
    },
    sendMessage(e) {
      e.preventDefault();

      // 计算当前时间并按格式输出
      this.timeCal();
      let value = this.textarea;
      this.send(value);
    },
    timeCal() {
      // 获取时间并按格式拼接
      let date = new Date();
      let year = date.getFullYear();
      let month = date.getMonth() + 1;
      let day = date.getDate();
      let hour = date.getHours();
      let minute = date.getMinutes();

      // 小于10的小时和分钟都在前面加上0
      if (hour < 10) {
        hour = "0" + hour;
      }
      if (minute < 10) {
        minute = "0" + minute;
      }

      let date_format =
        hour + ":" + minute + " " + month + "/" + day + "/" + year;

      this.currentTime = date_format;
    },
    send(value) {
      if (value == "") {
        this.$message({
          message: "The sent message is empty!",
          duration: 1500,
          type: "warning",
        });

        return;
      }

      if (sessionStorage.getItem("is_wait") == "false") {
        let that = this;

        // 计算当前时间并按格式输出
        this.timeCal();

        // 构建聊天记录对象并插入记录数组中
        let obj = {
          id: 1,
          nickName: "User",
          contactText: value,
          headUrl: "user.png",
          time: this.currentTime,
        };
        this.recordContent.push(obj);
        let question = value;
        this.textarea = "";

        let chatbot_obj = {
          id: 2,
          nickName: "Chatbot",
          contactText: "None",
          headUrl: "chatbot.png",
          time: that.currentTime,
          origin: 0,
        };

        this.recordContent.push(chatbot_obj);

        sessionStorage.setItem("is_wait", true);

        let is_stay_rule = sessionStorage.getItem("is_stay_rule");
        let type = sessionStorage.getItem("type");
        this.$axios
          .post("http://127.0.0.1:10000/getAnswer", {
            question: question,
            is_stay_rule: is_stay_rule,
            type: type,
          })
          .then((res) => {
            let len = that.recordContent.length;
            that.recordContent[len - 1].contactText = res.data.answer;
            that.recordContent[len - 1].origin = res.data.origin;

            sessionStorage.setItem("is_stay_rule", res.data.is_stay_rule);
            sessionStorage.setItem("type", res.data.type);
            sessionStorage.setItem("is_wait", false);

            if ("related_entity" in res.data) {
              that.ruleContent = res.data.related_entity;
            }
          })
          .catch(function (error) {
            let len = that.recordContent.length;
            that.recordContent[len - 1].contactText = "Sorry, Server Error!";

            sessionStorage.setItem("is_wait", false);
            sessionStorage.setItem("is_stay_rule", false);
            sessionStorage.setItem("type", "");

            // 请求失败处理
            that.$message({
              message: "Server Error!",
              duration: 1500,
              type: "error",
            });
          });
      } else {
        this.$message({
          message:
            "The last question has not been answered. PLease wait a moment!",
          duration: 1500,
          type: "warning",
        });
      }
    },
    sendTips(value, e) {
      e.preventDefault();
      this.send(value);
    },
    isShowKG(childValue) {
      this.$emit("showKG");
    },
    isShowMap(childValue) {
      this.$emit("showMap", childValue);
    },
    isShowRuleTips() {
      let value = this.ruleContent;
      this.showRuleTip = true;
      // this.$emit("showRuleTips", value);
    },
    closeRuleTip() {
      this.showRuleTip = false;
    },
    left_scroll() {
      let scroll_elem = document.querySelector("#scroll-tips");
      if (scroll_elem.scrollLeft != 0) {
        let scroll_elem = this.$refs.scroll_elem;
        scroll_elem.scrollTo({
          left: scroll_elem.scrollLeft - 200,
          behavior: "smooth",
        });
      }
    },
    right_scroll() {
      let scroll_elem = document.querySelector("#scroll-tips");
      if (
        scroll_elem.scrollLeft + scroll_elem.clientWidth <
        scroll_elem.scrollWidth
      ) {
        let scroll_elem = this.$refs.scroll_elem;
        scroll_elem.scrollTo({
          left: scroll_elem.scrollLeft + 200,
          behavior: "smooth",
        });
      }
    },
    faqSend(faq) {
      this.send(faq);
    },
    sendEntity(entity){
      this.showRuleTip = false;
      this.send(entity);
    }
  },
  computed: {
    filterEntities() {
      const { entity_seach_input, ruleContent, orderType } = this;
      let filterArr = new Array();

      // 过滤数组
      filterArr = ruleContent.filter((p) => p.indexOf(entity_seach_input) !== -1);

      return filterArr;
    },
  },
  watch: {
    recordContent: {
      handler() {
        // nextTick()，是将回调函数延迟在下一次dom更新数据后调用，简单的理解是：当数据更新了，在dom中渲染后，自动执行该函数
        this.$nextTick(() => {
          let container = this.$refs.scroll;
          // console.log(container.clientHeight + container.scrollTop);
          container.scrollTo({
            top: container.scrollHeight,
            behavior: "smooth",
          });
        });
      },
      deep: true,
    },
  },
  beforeMount() {
    this.timeCal();

    let obj = {
      id: 2,
      nickName: "Chatbot",
      contactText:
        "Intelligent assistant is at your service. How can I help you?",
      headUrl: "chatbot.png",
      time: this.currentTime,
      origin: 0,
    };

    this.recordContent.push(obj);
  },
  created() {
    sessionStorage.setItem("is_stay_rule", false);
    sessionStorage.setItem("type", "");
    sessionStorage.setItem("is_wait", false);

    let that = this;
    this.$axios.get("http://127.0.0.1:10000/getTips").then((res) => {
      that.faqList = res.data.faq;
      that.tipsList = res.data.tips;
    });
  },
};
</script>

<style lang="less" scoped>
#chat-root {
  width: 100%;
  height: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: row;
  position: relative;
  // background: #f3f3f3;
  #chat-left {
    width: 75%;
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
    box-shadow: inset 0px 2px 118px #a4d3ffcd;
    border-radius: 5px;
    --gray-2: rgba(0, 0, 0, 0.54);
    #chat-nav {
      font-weight: 500;
      color: white;
      font-weight: 900;
      p {
        text-align: center;
        margin: 10px 0;
      }
    }
    #chat-content {
      box-sizing: border-box;
      // width: 100%;
      padding: 8px;
      height: 100%;
      overflow: auto;
      margin-top: 10px;

      #chat-tips {
        width: auto;
        height: auto;
        display: block;
        color: var(--gray-2);
        font-size: 12px;
      }

      .word {
        display: flex;
        margin-bottom: 10px;

        img {
          width: 40px;
          height: 40px;
          border-radius: 50%;
        }
        .info {
          width: 47%;
          margin-left: 10px;
          text-align: left;
          .nickname {
            color: white;
            font-weight: 900;
          }
          .message_time {
            color: white;
            font-size: 12px;
            margin: 0;
            height: 20px;
            line-height: 20px;
            margin-top: -5px;
            margin-top: 5px;
          }
          .info-content {
            white-space: pre-line;
            word-wrap: break-word;
            // max-width: 45%;
            display: inline-block;
            padding: 10px;
            font-size: 14px;
            background: #fff;
            position: relative;
            margin-top: 8px;
            border-radius: 4px;
          }

          //小三角形
          .info-content::before {
            position: absolute;
            left: -8px;
            top: 8px;
            content: "";
            border-right: 10px solid #fff;
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
          }
        }
      }
      .info-first {
        animation-delay: 0.7s !important;
        animation-duration: 0.5s !important;
      }
      .info-content-first {
        animation-delay: 1.1s !important;
        animation-duration: 0.5s !important;
      }

      .info-others {
        animation-duration: 0.5s;
      }
      .info-content-others {
        animation-delay: 0.4s;
        animation-duration: 0.5s;
      }

      .word-my {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 10px;
        img {
          width: 60px;
          height: 60px;
          border-radius: 50%;
        }
        .info {
          width: 90%;
          // margin-left: 10px;
          text-align: right;
          // position: relative;
          display: flex;
          align-items: flex-end;
          flex-wrap: wrap;
          flex-direction: column;
          .nickname {
            margin-right: 10px;
            color: white;
            font-weight: 900;
          }
          .info-content {
            word-wrap: break-word;
            max-width: 45%;
            padding: 10px;
            font-size: 14px;
            margin-right: 10px;
            position: relative;
            margin-top: 8px;
            background: #97e24c;
            text-align: left;
            border-radius: 4px;
          }
          .Sender_time {
            color: white;
            padding-right: 12px;
            padding-top: 5px;
            font-size: 12px;
            margin: 0;
            height: 20px;
          }
          //小三角形
          .info-content::after {
            position: absolute;
            right: -8px;
            top: 8px;
            content: "";
            border-left: 10px solid #97e24c;
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
          }
        }
      }
    }
    #chat-content::-webkit-scrollbar {
      display: none;
    }
    #chat-scroll {
      display: flex;
      flex-direction: row;
      padding: 0 5px;
      .chat-scroll-button {
        text-align: center;
        line-height: 50px;
        img {
          vertical-align: middle;
          cursor: pointer;
          height: 30px;
        }
      }
      ul {
        overflow: hidden;
        width: 100%;
        margin: 0;
        padding: 0;
        white-space: nowrap;
        li {
          display: inline-block;
          // float: left;
          list-style-type: none;
          padding: 5px;
        }
      }
      ul:after {
        content: "";
        display: block;
        clear: both;
      }
    }
    #chat-input {
      text-align: right;
      flex-grow: 0;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      // border-top: 1px solid #c7c7c7;
      padding: 10px 10px;
      .chat-input-box {
        box-sizing: border-box;
        flex: 1 1 auto;
        padding-right: 10px;
      }
      .chat-input-button {
        display: inline-block;
        box-sizing: border-box;
        width: auto;
        align-self: flex-end;
      }
    }
  }
  #chat-right {
    // flex-grow: 1;
    border-radius: 5px;
    overflow: hidden;
    padding: 5px;
    width: 300px;
    #chat-card {
      box-shadow: inset 0px 2px 118px #ad9c9caf;
      overflow: hidden;
      border-radius: 10px;
      #chat-item-list {
        text-align: left;

        .content_scroll {
          overflow-y: scroll;
          height: 260px;
          width: 320px;
        }

        .text {
          font-size: 14px;
          font-weight: 500;
          max-width: 250px;
          white-space: nowrap;
          text-overflow: ellipsis;
          overflow: hidden;
          cursor: pointer;
        }

        .text:hover {
          font-weight: 900;
        }

        .contact {
          font-size: 14px;
        }

        .item {
          margin-bottom: 10px;
        }
        .clearfix {
          font-weight: 900;
          font-family: "Arial Black", "lcd", Impact, sans-serif;
        }
        .clearfix:before,
        .clearfix:after {
          display: table;
          content: "";
        }
        .clearfix:after {
          clear: both;
        }

        .box-card {
          max-width: 300px;
          background: none;
          color: white;
          border: 0;
        }
      }
      .chat-item-list-top {
        height: 360px;
      }
      .chat-item-list-bottom {
        height: 200px;
      }
    }

    .chat-contact {
      margin-top: 10px;
    }
  }
  #chat-entities {
    position: absolute;
    // background: red;
    width: 100%;
    height: 100%;
    border-radius: 5px;
    display: flex;
    justify-content: center;
    align-items: center;
    background: rgba(229, 229, 229, 0.441);
    #chat-entities-box {
      width: 80%;
      height: 80%;
      display: flex;
      flex-direction: column;
      background: #fff;
      #chat-entities-box-search {
        width: 100%;
      }
      #chat-entities-box-list {
        flex-grow: 1;
        overflow: hidden;
        div:first-child {
          height: 100%;
          width: calc(100% + 30px);
          overflow-y: scroll;
          ul {
            padding: 0;
            margin: 0;
            li {
              list-style: none;
              padding: 10px;
              background: #fff;
              border-bottom: 1px solid rgb(122, 122, 122);
              cursor: pointer;
            }
            li:hover {
              background: #41a2ff;
            }
          }
        }
      }
      #chat-entities-box-close {
        width: 100%;
        height: 40px;
        font-size: 25px;
        font-weight: 700;
        background: red;
        line-height: 40px;
        color: #fff;
        cursor: pointer;
      }
      #chat-entities-box-close:hover {
        background: #41a2ff;
        color: #fff;
      }
    }
  }

  .wait {
    width: 100px;
    display: flex;
    height: 20px;
    justify-content: center;
    align-items: center;
  }

  .wait div {
    margin: 5px;
    width: 20px;
    height: 10px;
    border-radius: 10px;
    background-color: lightblue;
    animation: wait 1.35s infinite;
  }

  @keyframes wait {
    50% {
      height: 20px;
    }
  }

  .wait div:nth-child(1) {
    animation-delay: 0.1s;
  }

  .wait div:nth-child(2) {
    animation-delay: 0.3s;
  }

  .wait div:nth-child(3) {
    animation-delay: 0.6s;
  }

  .wait div:nth-child(4) {
    animation-delay: 0.9s;
  }

  .wait div:nth-child(5) {
    animation-delay: 1.2s;
  }
}
</style>