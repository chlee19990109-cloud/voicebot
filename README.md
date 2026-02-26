# 음성 비서 프로그램 (Voice Bot Program)

**작성자**: 이충환
**서비스 링크**: [음성 비서 프로그램 바로가기](https://voicebot-ftudatsgedhwadz5be3pzt.streamlit.app/)

본 프로젝트는 OpenAI의 ChatGPT API(Whisper, GPT 모델)와 Google의 gTTS를 활용하여, 텍스트 입력 없이 사용자의 음성을 인식하고 음성으로 답변을 제공하는 **대화형 음성 비서(Voice Assistant)** 웹 애플리케이션입니다.

---

## 🚀 주요 기능 (Key Features)

- **UI 프레임워크**: Streamlit을 활용하여 직관적이고 반응형인 웹 인터페이스 제공.
- **STT (Speech-To-Text)**: 사용자의 음성을 녹음하고 OpenAI의 **Whisper AI** 모델을 활용하여 텍스트로 변환.
- **자연어 처리 및 대화 생성**: 사용자의 텍스트 질문을 OpenAI의 **GPT-4** 또는 **GPT-3.5-turbo** 모델에 전달하여 25단어 이내의 간결한 한국어 답변 생성 (이전 대화 맥락 유지 기능 포함).
- **TTS (Text-To-Speech)**: 생성된 텍스트 답변을 **Google Translate TTS(gTTS)** 를 통해 음성 파일로 변환하여 브라우저에서 자동 재생.

---

## 🛠️ 기술 요소 (Tech Stack)

### **프로그래밍 언어 및 프레임워크**
- **Python (3.13 사용)**: 프로젝트의 기반이 되는 프로그래밍 언어입니다. OpenAI의 인공지능 모델 API 호출 및 백엔드 로직 처리에 사용됩니다.
- **[Streamlit](https://streamlit.io/) (v1.52.2)**: 파이썬만으로 빠르게 데이터 분석용 및 AI 챗봇용 웹 UI를 구축할 수 있게 해주는 프레임워크입니다. 본 프로젝트의 전체 웹 인터페이스(화면 분할, 라디오 버튼 설정, 상태 저장 등)를 구성하는 데 사용되었습니다.

### **핵심 라이브러리 및 API**
- **[OpenAI API](https://openai.com/api/) (v0.28.1)**: 인공지능 처리의 핵심 요소입니다.
  - `whisper-1` 모델: 사용자가 마이크를 통해 녹음한 음성 데이터(mp3)를 입력받아, 높은 정확도로 텍스트 번역 및 인식을 수행하는 STT 모델입니다. 한국어 인식에 탁월한 성능을 보입니다.
  - `gpt-4` / `gpt-3.5-turbo` 모델: 텍스트로 변환된 사용자의 질문을 입력받아, 프롬프트 엔지니어링 설정에 따라 자연스럽고 논리적인 대답(텍스트)을 생성하는 LLM(대규모 언어 모델)입니다.
- **`gTTS` (Google Text-to-Speech) (v2.5.4)**: 구글의 번역 음성합성 엔진을 활용하여, GPT 모델이 생성한 텍스트 텍스트 답변을 사람의 육성과 유사한 한국어 음성 파일로 변환합니다. 웹페이지에서 바로 재생되도록 지원합니다.
- **`streamlit-audiorecorder` (v0.0.6)**: Streamlit 화면 상에 녹음 버튼 UI를 생성하고, 사용자의 웹브라우저 마이크를 통해 실시간으로 음성을 녹음하여 파이썬 코드로 데이터를 전달하는 확장 컴포넌트입니다.
- **`pydub` (v0.25.1)**: 오디오 파일의 구조를 다루고 확장자를 변환하는 라이브러리입니다. `streamlit-audiorecorder`로 수집된 날것의 음성 데이터를 Whisper AI가 처리할 수 있는 `mp3` 파일 포맷으로 변환하고 저장하는데 사용됩니다.

### **시스템 요구사항**
- **FFmpeg**: 디지털 오디오, 비디오를 인코딩 및 디코딩할 수 있는 시스템 레벨의 멀티미디어 프레임워크입니다. 파이썬 라이브러리인 `pydub`가 내부적으로 `ffmpeg`을 호출하여 오디오 데이터 처리를 수행하므로, 파이썬 환경과는 별개로 OS 자체에 반드시 설치되어 있어야 합니다. (웹호스팅 플랫폼인 Streamlit Cloud 배포 시에도 `packages.txt`에 명시되어 리눅스 환경에 설치되도록 설정해야 합니다.)

---

## 💻 구체적인 환경 설정 가이드 (Environment Setup)

### 1. 시스템 요구사항 설치: FFmpeg
이 애플리케이션의 핵심인 오디오 포맷 변환(`pydub`)을 위해 OS 상에 FFmpeg 설치가 선행되어야 합니다. FFmpeg이 없으면 음성 파일 처리에서 에러가 발생합니다.

- **Windows**: 
  1. [FFmpeg 공식 다운로드 페이지 (Windows 빌드)](https://github.com/BtbN/FFmpeg-Builds/releases)에서 `.zip` 파일을 다운로드하여 압축을 풉니다.
  2. 압축 해제된 폴더 안의 `bin` 폴더(예: `C:\ffmpeg\bin`)의 절대 경로를 복사합니다.
  3. 윈도우 검색창에 "환경 변수 편집"을 검색하여 실행합니다.
  4. '시스템 변수' (또는 '사용자 변수') 목록에서 `Path`를 선택하고 [편집]을 누릅니다.
  5. [새로 만들기]를 눌러 복사해둔 `bin` 폴더 경로를 추가하고 확인을 누릅니다.
  6. 명령 프롬프트(cmd)를 껐다가 켠 후 `ffmpeg -version`을 입력했을 때 버전 정보가 나오면 성공입니다.
- **macOS (Homebrew 사용)**: 
  1. 터미널을 열고 `brew install ffmpeg` 명령어를 실행합니다.
- **Linux (Ubuntu/Debian)**: 
  1. 터미널을 열고 `sudo apt-get update` 후 `sudo apt-get install ffmpeg` 명령어를 실행합니다.

### 2. 프로젝트 폴더 생성 및 소스 준비
명령 프롬프트(cmd) 또는 터미널을 실행하고 새로운 프로젝트 공간을 만듭니다.
```bash
# 최상위 경로(예: C 드라이브)에 프로젝트 디렉터리 생성 후 이동
cd \
mkdir chat-gpt-prg
cd chat-gpt-prg

# 실습 코드를 모아둘 폴더 생성 후 이동
mkdir ch03
cd ch03
```
준비된 파이썬 스크립트(`ch03_voicebot.py`), `requirements.txt`, `packages.txt` 파일을 해당 `ch03` 폴더 내로 복사하거나 다운로드 합니다.

### 3. Python 가상 환경(Virtual Environment) 생성 및 활성화
글로벌 파이썬 환경의 패키지 충돌을 막기 위해 현재 프로젝트 폴더 내에 단독으로 사용할 가상 환경을 구축합니다.

- **Windows 명령 프롬프트 (cmd)**:
  ```bash
  # ch03_env 라는 이름으로 파이썬 내장 venv 모듈을 이용해 가상환경 생성
  python -m venv ch03_env
  
  # 생성한 가상환경 활성화 스크립트 실행
  ch03_env\Scripts\activate.bat
  ```

- **macOS / Linux 터미널**:
  ```bash
  python3 -m venv ch03_env
  source ch03_env/bin/activate
  ```

*활성화가 완료되면 터미널 명령 줄 제일 앞에 `(ch03_env)`와 같이 가상환경 이름이 표시됩니다.*
**(주의!)** 만약 VS Code를 사용 중이라면 하단의 파이썬 인터프리터(Python Interpreter) 선택 메뉴에서 만들어둔 `ch03_env`를 수동으로 선택해 주어야 터미널에서도 정상 인식합니다.

### 4. 필요 라이브러리 일괄 설치
가상 환경이 활성화된 `(ch03_env)` 상태에서, 프로젝트에 필요한 모든 서드파티 라이브러리 명세서인 `requirements.txt`를 읽어들여 일괄 설치합니다.
```bash
# pip를 활용한 의존성 일괄 설치 의뢰
pip install -r requirements.txt
```
정상적으로 설치가 진행되면 `streamlit`, `openai`, `gTTS` 등의 패키지가 다운로드되어 `ch03_env` 환경 내에 저장됩니다.

---

## ▶️ 실행 가이드 (How to Run)

모든 환경 설정이 완료되었다면 다음 명령어를 통해 Streamlit 애플리케이션을 실행합니다.

```bash
streamlit run ch03_voicebot.py
```

1. 실행 후 자동으로 기본 브라우저가 열리며 애플리케이션 화면(`http://localhost:8501`)에 접속됩니다.
2. 좌측 사이드바(Sidebar) 란에 본인의 **[OpenAI API Key](https://platform.openai.com/account/api-keys)**를 입력합니다.
3. 사용할 GPT 모델(GPT-4 또는 GPT-3.5-turbo)을 선택합니다.
4. **"클릭하여 녹음하기"** 버튼을 눌러 음성으로 질문하면 챗봇이 음성으로 대답합니다.

---

## ⚠️ 사용 시 주의사항
- **API 키 관리**: 소스코드에 API 키를 하드코딩하지 않았으나, 깃허브 등에 배포 시 API 키가 노출되지 않도록 각별히 유의해야 합니다.
- **OpenAI 라이브러리 버전**: 본 프로젝트는 구버전 문법인 `openai==0.28.1`에 맞추어 작성되어 있습니다. 만약 최신 버전(1.x)의 `openai` 라이브러리로 강제 업데이트될 경우, 코드 내 함수 호출 방식(`openai.ChatCompletion.create` 등)의 수정이 필요할 수 있습니다.
- **의존성 충돌**: 시스템 내 다른 파이썬 버전이나 패키지와의 충돌 방지를 위해 **반드시 가상 환경(`venv`) 내에서 실행**해주세요.
