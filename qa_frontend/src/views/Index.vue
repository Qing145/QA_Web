<template>
  <div id="index-root">
    <div id="root-box" class="animate__animated animate__fadeInDown">
      <div id="root-box-text">QUESTION ANSWERING SYSTEM - POLYU</div>
      <div id="root-box-button" @click="enter">
        Enter System
      </div>
    </div>
  </div>
</template>
<script>
export default {
  name: "index",
  data() {
    return {
    };
  },
  methods: {
    enter(){
        this.$router.push('/chatbot');
    }
  },
};
</script>
<style lang="less" scoped>
#index-root {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  #root-box {
    width: 80%;
    height: 100%;
    box-shadow: inset 0 2px 118px #41a2ff;
    border: 1px solid #41a2ff;
    border-radius: 20px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    #root-box-text {
      width: 100%;
      line-height: 120px;
      text-shadow: 0 2px 15px #41a2ff;
      font-family: digifaceWide, lcd, Impact, Arial Black, sans-serif;
      font-size: 40px;
      font-weight: 700;
      color: #fff;
      overflow: hidden;
      background-image: url("~@/assets/xiantiao_07.png");
    }

    #root-box-button{
        border: 1px solid #41a2ff;
        color: #fff;
        font-size: 24px;
        height: 60px;
        line-height: 60px;
        width: 300px;
        cursor: pointer;
        margin-top: 60px;
        border-radius: 25px;
    }

    #root-box-button:hover{
        background: #fff;
        color: #41a2ff;
    }
  }

  .animate__animated.animate__fadeInDown {
    --animate-duration: 1s;
  }
}
</style>