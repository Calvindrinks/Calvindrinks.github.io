### safety.h

```cpp
/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
 PowerProtect is mainly power protection, using the principle of p=fv, "torque * angular velocity", that is, the sum of 12 motor torque times their angular velocity, evenly divided into 1-10 levels according to the empirical value, can be slowly added when using. (Experience value, total power 1000w, 5th gear is 500w)
Power protection, using hand to shake the leg can also passively triggers protection.
If the function returns -1 after triggered, The function does not modify the command.
***********************************************************************/

#ifndef _UNITREE_LEGGED_SAFETY_H_
#define _UNITREE_LEGGED_SAFETY_H_

#include "comm.h"
#include "quadruped.h"

namespace UNITREE_LEGGED_SDK
{

  class Safety
  {
  public:
    Safety(LeggedType type);
    ~Safety();
    void PositionLimit(LowCmd &);                                    // limit the max command, only effect under Low Level control in Position mode
    int PowerProtect(LowCmd &, LowState &, int);                     /* only effect under Low Level control, input factor: 1~10,means 10%~100% power limit. 
    																	If you are new, then use 1; if you are familiar,then can try bigger number or even comment this function. */
    int PositionProtect(LowCmd &, LowState &, double limit = 0.087); // default limit is 5 degree
  private:
    int WattLimit, Wcount; // Watt. When limit to 100, you can triger it with 4 hands shaking.
    double Hip_max, Hip_min, Thigh_max, Thigh_min, Calf_max, Calf_min;
  };

} // namespace UNITREE_LEGGED_SDK

#endif
```



### go1_const.h

[constexpr](https://learn.microsoft.com/en-us/cpp/cpp/constexpr-cpp?view=msvc-170)

The keyword **`constexpr`** was introduced in C++11 and improved in C++14. It means *constant expression*. Like **`const`**, it can be applied to variables: A compiler error is raised when any code attempts to modify the value. Unlike **`const`**, **`constexpr`** can also be applied to functions and class constructors. **`constexpr`** indicates that the value, or return value, is constant and, where possible, is computed at compile time.

```cpp
/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#ifndef _UNITREE_LEGGED_GO1_H_
#define _UNITREE_LEGGED_GO1_H_

namespace UNITREE_LEGGED_SDK {
constexpr double go1_Hip_max = 1.047;     // unit:radian ( = 60   degree)
constexpr double go1_Hip_min = -1.047;    // unit:radian ( = -60  degree)
constexpr double go1_Thigh_max = 2.966;   // unit:radian ( = 170  degree)
constexpr double go1_Thigh_min = -0.663;  // unit:radian ( = -38  degree)
constexpr double go1_Calf_max = -0.837;   // unit:radian ( = -48  degree)
constexpr double go1_Calf_min = -2.721;   // unit:radian ( = -156 degree)
}  // namespace UNITREE_LEGGED_SDK

#endif
```



### quadruped.h

[enum class](https://learn.microsoft.com/en-us/cpp/cpp/constexpr-cpp?view=msvc-170)

```cpp
/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#ifndef _UNITREE_LEGGED_QUADRUPED_H_
#define _UNITREE_LEGGED_QUADRUPED_H_

#include <string>

using namespace std;

namespace UNITREE_LEGGED_SDK {

enum class LeggedType {
  Aliengo,
  A1,
  Go1,
  B1
};

string VersionSDK();
int InitEnvironment();  // memory lock

// definition of each leg and joint
constexpr int FR_ = 0;  // leg index
constexpr int FL_ = 1;
constexpr int RR_ = 2;
constexpr int RL_ = 3;

constexpr int FR_0 = 0;  // joint index
constexpr int FR_1 = 1;
constexpr int FR_2 = 2;

constexpr int FL_0 = 3;
constexpr int FL_1 = 4;
constexpr int FL_2 = 5;

constexpr int RR_0 = 6;
constexpr int RR_1 = 7;
constexpr int RR_2 = 8;

constexpr int RL_0 = 9;
constexpr int RL_1 = 10;
constexpr int RL_2 = 11;

}  // namespace UNITREE_LEGGED_SDK

#endif
```

## comm.h
### LowCmd

```cpp
typedef struct
{
  std::array<uint8_t, 2> head;
  uint8_t levelFlag;
  uint8_t frameReserve;
  std::array<uint32_t, 2> SN;
  std::array<uint32_t, 2> version;
  uint16_t bandWidth;
  std::array<MotorCmd, 20> motorCmd;       // same index as quadruped.h
  BmsCmd bms;
  std::array<uint8_t, 40> wirelessRemote;  // wireless commands
  uint32_t reserve;
  uint32_t crc;
} LowCmd;  // low level control
```

### lowstate

```cpp
typedef struct
{
  std::array<uint8_t, 2> head;
  uint8_t levelFlag;
  uint8_t frameReserve;
  std::array<uint32_t, 2> SN;
  std::array<uint32_t, 2> version;
  uint16_t bandWidth;
  IMU imu;
  std::array<MotorState, 20> motorState;   // same index as quadruped.h
  BmsState bms;
  std::array<int16_t, 4> footForce;        // force sensors value, ground detection
  std::array<int16_t, 4> footForceEst;     // force estimation sensors by current (unit: N)
  uint32_t tick;                           // reference real-time from motion controller (unit: ms)
  std::array<uint8_t, 40> wirelessRemote;  // wireless commands
  uint32_t reserve;
  uint32_t crc;
} LowState;  // low level feedback
```

### motercmd

```cpp
  typedef struct
  {
    uint8_t mode; // desired working mode
    float q;      // desired angle (unit: radian)
    float dq;     // desired velocity (unit: radian/second)
    float tau;    // desired output torque (unit: N.m)
    float Kp;     // desired position stiffness (unit: N.m/rad )
    float Kd;     // desired velocity stiffness (unit: N.m/(rad/s) )
    std::array<uint32_t, 3> reserve;
  } MotorCmd; // motor control
```

### motorstate

```cpp
typedef struct
{
  uint8_t mode;  // motor working mode
  float q;       // current angle (unit: radian)
  float dq;      // current velocity (unit: radian/second)
  float ddq;     // current acc (unit: radian/second*second)
  float tauEst;  // current estimated output torque (unit: N.m)
  float q_raw;   // current raw angle (unit: radian)j
  float dq_raw;  // current raw velocity (unit: radian/second)
  float ddq_raw; // current raw acc (unit: radian/second)
  int8_t temperature;  // current temperature (temperature conduction is slow that leads to lag)
  std::array<uint32_t, 2> reserve;
} MotorState;  // motor feedback
```

### imu

```cpp
typedef struct
{
  std::array<float, 4> quaternion;     // quaternion, normalized, (w,x,y,z)
  std::array<float, 3> gyroscope;      // angular velocity, (w,x,y) （unit: rad/s)    (raw data)
  std::array<float, 3> accelerometer;  // m/(s2)      (raw data)
  std::array<float, 3> rpy;            // euler angle（unit: rad)
  int8_t temperature;
} IMU;  // when under accelerated motion, the attitude of the robot calculated by IMU will drift.
```

