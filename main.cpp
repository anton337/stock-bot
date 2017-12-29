#include <iostream>
#include <vector>
#include <stdlib.h>
#include <sstream>
#include <stdio.h>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
#include <math.h>
#include <set>

/*
 
 RSI Oversold in Uptrend

 This scan reveals stocks that are in an uptrend with oversold RSI. First, stocks must be above their 200-day moving average to be in an overall uptrend. Second, RSI must cross below 30 to become oversold.

 [type = stock] AND [country = US] 
 AND [Daily SMA(20,Daily Volume) > 40000] 
 AND [Daily SMA(60,Daily Close) > 20] 

 AND [Daily Close > Daily SMA(200,Daily Close)] 
 AND [Daily RSI(5,Daily Close) <= 30]

 RSI Overbought in Downtrend

 This scan reveals stocks that are in a downtrend with overbought RSI turning down. First, stocks must be below their 200-day moving average to be in an overall downtrend. Second, RSI must cross above 70 to become overbought.

 [type = stock] AND [country = US] 
 AND [Daily SMA(20,Daily Volume) > 40000] 
 AND [Daily SMA(60,Daily Close) > 20] 

 AND [Daily Close < Daily SMA(200,Daily Close)] 
 AND [Daily RSI(5,Daily Close) >= 70]

*/

float max(float a,float b)
{
  return (a>b)?a:b;
}

float min(float a,float b)
{
  return (a>b)?b:a;
}

float fabs(float a)
{
  return (a>0)?a:-a;
}

struct Point
{
  float t;
  float x,y;
  Point(float _x,float _y):x(_x),y(_y){}
  float dist(Point const & a,float alpha)
  {
    return pow(sqrt((x-a.x)*(x-a.x) + (y-a.y)*(y-a.y)),alpha);
  }
};

float total_dist(std::vector<Point> & pts,float alpha = 1)
{
  float temp_dist;
  float total_dist = 0;
  pts[0].t = total_dist;
  for(int i=1;i<pts.size();i++)
  {
    temp_dist = pts[i].dist(pts[i-1],alpha);
    total_dist += temp_dist;
    pts[i].t = total_dist;
  }
}

int find(std::vector<Point> & pts,float t)
{
  for(int i=1;i<pts.size();i++)
  {
    if(pts[i].t > t)return i-1;
  }
}

Point CatmulRom(std::vector<Point> & pts,float t)
{
  int ind0 = (int)find(pts,t)-1;
  int ind1 = (int)find(pts,t);
  int ind2 = (int)find(pts,t)+1;
  int ind3 = (int)find(pts,t)+2;
  //std::cout << pts[ind1].t << "\t" << pts[ind1].x << "\t" << pts[ind1].y << std::endl;
  //std::cout << "$$$$$" << ind1 << std::endl;
  if(ind0<0)ind0 = 0;
  if(ind1<0)ind1 = 0;
  if(ind2<0)ind2 = 0;
  if(ind3<0)ind3 = 0;
  if(ind0>=pts.size())ind0 = pts.size()-1;
  if(ind1>=pts.size())ind1 = pts.size()-1;
  if(ind2>=pts.size())ind2 = pts.size()-1;
  if(ind3>=pts.size())ind3 = pts.size()-1;
  //std::cout << t << "~~~~" << pts[ind0].t << '\t' << pts[ind1].t << '\t' << pts[ind2].t << '\t' << pts[ind3].t << std::endl;
  //std::cout << "^^^^" << pts[ind0].x << '\t' << pts[ind1].x << '\t' << pts[ind2].x << '\t' << pts[ind3].x << std::endl;
  float d10 = pts[ind1].t - pts[ind0].t;
  float d21 = pts[ind2].t - pts[ind1].t;
  float d32 = pts[ind3].t - pts[ind2].t;
  float A1x = (d10>1e-5)?(pts[ind0].x*(pts[ind1].t-t) + pts[ind1].x*(t-pts[ind0].t))/d10:pts[ind0].x;
  float A1y = (d10>1e-5)?(pts[ind0].y*(pts[ind1].t-t) + pts[ind1].y*(t-pts[ind0].t))/d10:pts[ind0].y;
  float A2x = (d21>1e-5)?(pts[ind1].x*(pts[ind2].t-t) + pts[ind2].x*(t-pts[ind1].t))/d21:pts[ind1].x;
  float A2y = (d21>1e-5)?(pts[ind1].y*(pts[ind2].t-t) + pts[ind2].y*(t-pts[ind1].t))/d21:pts[ind1].y;
  float A3x = (d32>1e-5)?(pts[ind2].x*(pts[ind3].t-t) + pts[ind3].x*(t-pts[ind2].t))/d32:pts[ind2].x;
  float A3y = (d32>1e-5)?(pts[ind2].y*(pts[ind3].t-t) + pts[ind3].y*(t-pts[ind2].t))/d32:pts[ind2].y;
  //std::cout << "^^^" << A1x << '\t' << A2x << '\t' << A3x << std::endl;
  float d20 = pts[ind2].t - pts[ind0].t;
  float d31 = pts[ind3].t - pts[ind1].t;
  float B1x = (d20>1e-5)?(A1x*(pts[ind2].t-t) + A2x*(t-pts[ind0].t))/d20:A1x;
  float B1y = (d20>1e-5)?(A1y*(pts[ind2].t-t) + A2y*(t-pts[ind0].t))/d20:A1y;
  float B2x = (d31>1e-5)?(A2x*(pts[ind3].t-t) + A3x*(t-pts[ind1].t))/d31:A2x;
  float B2y = (d31>1e-5)?(A2y*(pts[ind3].t-t) + A3y*(t-pts[ind1].t))/d31:A2y;
  //std::cout << "^^" << B1x << '\t' << B2x << std::endl;
  float Cx  = (d21>1e-5)?(B1x*(pts[ind2].t-t) + B2x*(t-pts[ind1].t))/d21:B1x;
  float Cy  = (d21>1e-5)?(B1y*(pts[ind2].t-t) + B2y*(t-pts[ind1].t))/d21:B1y;
  //std::cout << "^" << Cx << "\t" << Cy << std::endl;
  return Point(Cx,Cy);
}


Point estimate_derivative(std::vector<Point> & pts,float a,float dx)
{
  Point p1 = CatmulRom(pts,a-dx);
  Point p2 = CatmulRom(pts,a+dx);
  return Point((p2.x-p1.x)/(2*dx),(p2.y-p1.y)/(2*dx));
}

struct price
{

  price()
  {
    EMA = 0;
    EMA1 = 0;
    EMS = 0;
    EMS1 = 0;
  }

  // these quantities are ground truth
  int index;
  std::string date;
  float open;
  float close;
  float high;
  float low;
  int volume;

  // these quantities are derived from the values above
  
  float EMA_MACD; // temporary ema MACD
  float calculate_ema_macd(std::vector<price> & prices,int N)
  {
    float a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMA_MACD = a * (MACD_line - prices[index-1].EMA_MACD) + prices[index-1].EMA_MACD;
      return EMA_MACD;
    }
    else
    {
      EMA_MACD = MACD_line;
      return EMA_MACD;
    }
  }
  float EMA_MACD1; // temporary ema MACD
  float calculate_ema_macd1(std::vector<price> & prices,int N)
  {
    float a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMA_MACD1 = a * (MACD_line - prices[index-1].EMA_MACD1) + prices[index-1].EMA_MACD1;
      return EMA_MACD1;
    }
    else
    {
      EMA_MACD1 = MACD_line;
      return EMA_MACD1;
    }
  }
  float EMA; // temporary ema
  float calculate_ema(std::vector<price> & prices,int N)
  {
    float a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMA = a * (close - prices[index-1].EMA) + prices[index-1].EMA;
      return EMA;
    }
    else
    {
      EMA = close;
      return EMA;
    }
  }
  float EMA1; // temporary ema
  float calculate_ema1(std::vector<price> & prices,int N)
  {
    float a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMA1 = a * (close - prices[index-1].EMA1) + prices[index-1].EMA1;
      return EMA1;
    }
    else
    {
      EMA1 = close;
      return EMA1;
    }
  }
  float EMS; // temporary ems
  float calculate_ems(std::vector<price> & prices,float mean,int N)
  {
    float a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMS = a * ((close - mean)*(close - mean) - prices[index-1].EMS) + prices[index-1].EMS;
      return EMS;
    }
    else
    {
      EMS = 0;
      return EMS;
    }
  }
  float EMS1; // temporary ems
  float calculate_ems1(std::vector<price> & prices,float mean,int N)
  {
    float a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMS1 = a * ((close - mean)*(close - mean) - prices[index-1].EMS1) + prices[index-1].EMS1;
      return EMS1;
    }
    else
    {
      EMS1 = 0;
      return EMS1;
    }
  }
  float EMAV; // temporary ema
  float calculate_ema_volume(std::vector<price> & prices,int N)
  {
    float a = 2.0f / ( 1.0f + N );
    if(index>=N)
    {
      EMAV = a * (volume - prices[index-1].EMAV) + prices[index-1].EMAV;
      return EMAV;
    }
    else
    {
      EMAV = volume;
      return EMAV;
    }
  }
  float SMAV; // temporary simple moving average
  float calculate_sma_volume(std::vector<price> & prices,int N)
  {
    if(index>=N)
    {
      SMAV = 0;
      for(int i=1;i<=N;i++)
      {
        SMAV += (prices[index+1-i].volume - SMAV)/i;
      }
      return SMAV;
    }
    else
    {
      SMAV = 0;
      for(int i=1;i<=index+1;i++)
      {
        SMAV += (prices[index+1-i].volume - SMAV)/i;
      }
      return SMAV;
    }
  }

  // Volume Spike - sizzle index
  // Current Volume > tolerance x (5 d SMA Volume)
  // close today > close yesterday => volume gain
  // close today < close yesterday => volume loss
  bool Volume_spike;
  bool Volume_spike_gain;
  bool Volume_spike_loss;
  bool Volume_gain;
  bool Volume_loss;
  void calculate_Volume_spike(std::vector<price> & prices,int N=5,float tolerance=2.0f)
  {
    Volume_spike = volume > tolerance * SMAV;
    Volume_spike_gain = false;
    Volume_spike_loss = false;
    Volume_gain = false;
    Volume_loss = false;
    if(index >= 1)
    {
      if(Volume_spike)
      {
        if(close > prices[index-1].close)
        {
          Volume_spike_gain = true;
        }
        else
        {
          Volume_spike_loss = true;
        }
      }
      if(close > prices[index-1].close)
      {
        Volume_gain = true;
      }
      else
      {
        Volume_loss = true;
      }
    }
  }
  static void initialize_Volume_spike(std::vector<price> & prices, int N=5,float tolerance=2.0f)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_sma_volume(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_Volume_spike(prices,N,tolerance);
    }
  }

  // RSI 
  // 100 - 100/(1+RS)
  // RS = Average Gain / Average Loss
  float average_gain;
  float average_loss;
  float RS;
  float RSI;
  bool RSI_buy;
  bool RSI_sell;
  void calculate_RSI(std::vector<price> & prices,int N=14)
  {
    if(index==N)
    {
      average_gain = 0;
      average_loss = 0;
      for(int i=1;i<N+1;i++)
      {
        if(prices[index+1-i].close > prices[index-i].close)
        {
          average_gain += ((prices[index+1-i].close-prices[index-i].close) - average_gain)/i;
          average_loss -= average_loss/i;
        }
        else
        {
          average_loss += ((prices[index-i].close-prices[index+1-i].close) - average_loss)/i;
          average_gain -= average_gain/i;
        }
      }
      if(average_loss>1e-10)
      {
        RS = average_gain / average_loss;
      }
      else
      {
        RS = 0;
      }
      RSI = 100 - 100/(1+RS);
    }
    else
    if(index>=N+1)
    {
      if(prices[index].close > prices[index-1].close)
      {
        average_gain = ((prices[index].close-prices[index-1].close) + (N-1)*prices[index-1].average_gain)/N;
        average_loss = ((N-1)*prices[index-1].average_loss)/N;
      }
      else
      {
        average_loss = ((prices[index-1].close-prices[index].close) + (N-1)*prices[index-1].average_loss)/N;
        average_gain = ((N-1)*prices[index-1].average_gain)/N;
      }
      if(average_loss>1e-10)
      {
        RS = average_gain / average_loss;
      }
      else
      {
        RS = 0;
      }
      RSI = 100 - 100/(1+RS);
    }
    else
    {
      average_gain = 0;
      average_loss = 0;
      RS = 0;
      RSI = 0;
    }
    //std::cout << index << "\t" << date << "\t" << average_gain << "\t" << average_loss << "\t" << RS << "\t" << RSI << std::endl;
    //char ch;
    //std::cin >> ch;
  }
  void calculate_RSI_buy()
  {
    if(RSI>1e-10)
    {
      RSI_buy = RSI<30;
    }
    else
    {
      RSI_buy = false;
    }
  }
  void calculate_RSI_sell()
  {
    if(RSI>1e-10)
    {
      RSI_sell = RSI>70;
    }
    else
    {
      RSI_sell = false;
    }
  }
  static void initialize_RSI(std::vector<price> & prices,int N=14)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_RSI(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_RSI_buy();
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_RSI_sell();
    }
  }

  // MFI 
  // 100 - 100/(1+MF)
  // MF = Average TP Gain * Gain Volume / Average TP Loss * Loss Volume
  float raw_money_flow_gain;
  float raw_money_flow_loss;
  float MF;
  float MFI;
  bool MFI_buy;
  bool MFI_sell;
  void calculate_MFI(std::vector<price> & prices,int N=14)
  {
    if(index==N)
    {
      raw_money_flow_gain = 0;
      raw_money_flow_loss = 0;
      for(int i=1;i<N+1;i++)
      {
        if(prices[index+1-i].TP > prices[index-i].TP)
        {
          raw_money_flow_gain += (prices[index+1-i].TP*prices[index+1-i].volume - raw_money_flow_gain)/i;
          raw_money_flow_loss -= raw_money_flow_loss/i;
        }
        else
        {
          raw_money_flow_loss += (prices[index+1-i].TP*prices[index+1-i].volume - raw_money_flow_loss)/i;
          raw_money_flow_gain -= raw_money_flow_gain/i;
        }
      }
      if(raw_money_flow_loss>1e-10)
      {
        MF = raw_money_flow_gain / raw_money_flow_loss;
      }
      else
      {
        MF = 0;
      }
      MFI = 100 - 100/(1+MF);
    }
    else
    if(index>=N+1)
    {
      if(prices[index].TP > prices[index-1].TP)
      {
        raw_money_flow_gain = (prices[index].TP*prices[index-1].volume + (N-1)*prices[index-1].raw_money_flow_gain)/N;
        raw_money_flow_loss = ((N-1)*prices[index-1].raw_money_flow_loss)/N;
      }
      else
      {
        raw_money_flow_loss = (prices[index-1].TP*prices[index].volume + (N-1)*prices[index-1].raw_money_flow_loss)/N;
        raw_money_flow_gain = ((N-1)*prices[index-1].raw_money_flow_gain)/N;
      }
      if(raw_money_flow_loss>1e-10)
      {
        MF = raw_money_flow_gain / raw_money_flow_loss;
      }
      else
      {
        MF = 0;
      }
      MFI = 100 - 100/(1+MF);
    }
    else
    {
      raw_money_flow_gain = 0;
      raw_money_flow_loss = 0;
      MF = 0;
      MFI = 0;
    }
  }
  void calculate_MFI_buy()
  {
    if(MFI>1e-10)
    {
      MFI_buy = MFI<30;
    }
    else
    {
      MFI_buy = false;
    }
  }
  void calculate_MFI_sell()
  {
    if(MFI>1e-10)
    {
      MFI_sell = MFI>70;
    }
    else
    {
      MFI_sell = false;
    }
  }
  static void initialize_MFI(std::vector<price> & prices,int N=14)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_TP();
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MFI(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MFI_buy();
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MFI_sell();
    }
  }

  // Golden cross
  float ema_50;
  float ema_200;
  float ems_50;
  float ems_200;
  bool GoldenCross_uptrend;
  bool GoldenCross_downtrend;
  void calculate_GoldenCross_uptrend(std::vector<price> & prices,int N1=50,int N2=200)
  {
    if(index>=N2&&index>=N1)
    {
      ema_50 = calculate_ema(prices,N1);
      ema_200= calculate_ema1(prices,N2);
      ems_50 = sqrtf(calculate_ems(prices,ema_50,N1));
      ems_200= sqrtf(calculate_ems1(prices,ema_200,N2));
      GoldenCross_uptrend = ema_50>ema_200;
    }
    else
    {
      ema_50 = close;
      ema_200 = close;
      ems_50 = 0;
      ems_200 = 0;
      GoldenCross_uptrend = false;
    }
  }
  void calculate_GoldenCross_downtrend(std::vector<price> & prices,int N1=50,int N2=200)
  {
    if(index>=N2&&index>=N1)
    {
      ema_50 = calculate_ema(prices,N1);
      ema_200= calculate_ema1(prices,N2);
      ems_50 = sqrtf(calculate_ems(prices,ema_50,N1));
      ems_200= sqrtf(calculate_ems1(prices,ema_200,N2));
      GoldenCross_downtrend = ema_50<ema_200;
    }
    else
    {
      ema_50 = close;
      ema_200 = close;
      ems_50 = 0;
      ems_200 = 0;
      GoldenCross_downtrend = false;
    }
  }
  static void initialize_GoldenCross(std::vector<price> & prices,int N1=50,int N2=200)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_GoldenCross_uptrend(prices,N1,N2);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_GoldenCross_downtrend(prices,N1,N2);
    }
  }

  // MACD
  float ema_12;
  float ema_26;
  float ems_12;
  float ems_26;
  float MACD_line;
  float MACD_signal;
  float MACD_dline;
  float MACD_dsignal;
  bool MACD_uptrend;
  bool MACD_downtrend;
  void calculate_MACD_signal(std::vector<price> & prices,int N=9,int N1=12,int N2=26)
  {
    if(index>=N&&index>=N1&&index>=N2&&index>=100)
    {
      ema_12 = calculate_ema(prices,N1);
      ema_26 = calculate_ema1(prices,N2);
      if(index>=200)
      {
        MACD_line = (ema_12 - ema_26)/(ema_26+1);
        MACD_signal = calculate_ema_macd(prices,N);
      }
      else
      {
        MACD_line = 0;
        MACD_signal = 0;
      }
    }
    else
    {
      ema_12 = close;
      ema_26 = close;
      MACD_line = 0;
      MACD_signal = 0;
    }
  }
  static void calculate_MACD_dsignal(std::vector<price> & prices)
  {
    //char ch;
    //std::cout << "pass 1" << std::endl;
    std::vector<Point> pts;
    for(int i=0;i<prices.size();i++)
    {
      pts.push_back(Point(1000*prices[i].MACD_line,1000*prices[i].MACD_signal));
      //std::cout << pts[i].x << "\t" << pts[i].y << std::endl;
    }
    //std::cin >> ch;
    //std::cout << "pass 2" << std::endl;
    total_dist(pts,0.5);
    //total_dist(pts,0.0);
    //for(int i=0;i<prices.size();i++)
    //{
      //std::cout << pts[i].t << "\t" << pts[i].x << "\t" << pts[i].y << std::endl;
    //}
    //std::cout << "pass 3" << std::endl;
    //std::cin >> ch;
    prices[0].MACD_dline   = 0.0001;
    prices[0].MACD_dsignal = 0.0001;
    for(int i=0;i+1<pts.size();i++)
    {
      //std::cout << "i=" << i << "\t" << pts[i].t << std::endl;
      //Point pt = CatmulRom(pts,0.5f*(pts[i].t+pts[i+1].t));
      //Point drv = estimate_derivative(pts,0.5f*(pts[i].t+pts[i+1].t),0.01);
      Point drv = estimate_derivative(pts,pts[i].t,0.01);
      prices[i+1].MACD_dline = drv.x;
      prices[i+1].MACD_dsignal = drv.y;
      //prices[i+1].MACD_dline = pt.x;
      //prices[i+1].MACD_dsignal = pt.y;
      //prices[i+1].MACD_dline = pts[i].x;
      //prices[i+1].MACD_dsignal = pts[i].y;
    }
    //std::cin >> ch;
    //std::cout << "pass 4" << std::endl;
  }
  void calculate_MACD_uptrend(std::vector<price> & prices,int N1=12,int N2=26)
  {
    if(index>=N2&&index>=N1&&index>=100)
    {
      ema_12 = calculate_ema(prices,N1);
      ema_26 = calculate_ema1(prices,N2);
      ems_12 = sqrtf(calculate_ems(prices,ema_12,N1));
      ems_26 = sqrtf(calculate_ems1(prices,ema_26,N2));
      MACD_uptrend = ema_12>ema_26;
    }
    else
    {
      ema_12 = close;
      ema_26 = close;
      ems_12 = 0;
      ems_26 = 0;
      MACD_uptrend = false;
    }
  }
  void calculate_MACD_downtrend(std::vector<price> & prices,int N1=12,int N2=26)
  {
    if(index>=N2&&index>=N1&&index>=100)
    {
      ema_12 = calculate_ema(prices,N1);
      ema_26 = calculate_ema1(prices,N2);
      ems_12 = sqrtf(calculate_ems(prices,ema_12,N1));
      ems_26 = sqrtf(calculate_ems1(prices,ema_26,N2));
      MACD_downtrend = ema_12<ema_26;
    }
    else
    {
      ema_12 = close;
      ema_26 = close;
      ems_12 = 0;
      ems_26 = 0;
      MACD_downtrend = false;
    }
  }
  static void initialize_MACD(std::vector<price> & prices,int N=9,int N1=12,int N2=26)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MACD_uptrend(prices,N1,N2);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MACD_downtrend(prices,N1,N2);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MACD_signal(prices,N,N1,N2);
    }
    calculate_MACD_dsignal(prices);
  }

  // DOJI - open and close price are very similar
  bool doji;
  void calculate_doji()
  {
    doji = open + 0.003*open > close && open - 0.003*open < close;
  }
  static void initialize_doji(std::vector<price> & prices)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_doji();
    }
  }

  // CCI
  float TP; // typical price = (high + low + close)/3
  float smtp_cci; // 20 day simple moving average of Typical Price (TP)
  float MD_cci; // 20 day mean deviation = sum_n |TP - smtp_n|/n
  float CCI; // (TP - 20d SMTP) / (.015 MD)
  bool CCI_buy, CCI_sell;
  static void initialize_CCI(std::vector<price> & prices,int N = 20)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_TP();
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_smtp(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_MD(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_CCI(prices,N);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_CCI_buy();
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_CCI_sell();
    }
  }
  void calculate_TP()
  {
    TP = (high + low + close)/3.0f;
  }
  void calculate_smtp(std::vector<price> & prices,int N = 20)
  {
    if(index>=N)
    {
      smtp_cci = 0;
      for(int k=1;k<=N;k++)
      {
        smtp_cci += (prices[index+1-k].TP - smtp_cci)/k;
      }
    }
    else
    {
      smtp_cci = 0;
    }
  }
  void calculate_MD(std::vector<price> & prices,int N = 20)
  {
    if(index>=N)
    {
      MD_cci = 0;
      for(int k=1;k<=N;k++)
      {
        MD_cci += (fabs(prices[index+1-k].TP - smtp_cci) - MD_cci)/k;
      }
    }
    else
    {
      smtp_cci = 0;
    }
  }
  void calculate_CCI(std::vector<price> & prices,int N=20)
  {
    if(index>=N)
    {
      CCI = (TP - smtp_cci) / (.015 * MD_cci);
    }
    else
    {
      CCI = 0;
    }
  }
  void calculate_CCI_buy()
  {
    if(fabs(CCI-0)>1e-10)
    {
      CCI_buy = CCI < -100;
    }
    else
    {
      CCI_buy = false;
    }
  }
  void calculate_CCI_sell()
  {
    if(fabs(CCI-0)>1e-10)
    {
      CCI_sell = CCI > 100;
    }
    else
    {
      CCI_sell = false;
    }
  }

  // Bullish Engulfing Pattern
  // Today close > Yesterday open
  // Today open < Yesterday close
  // Today close > Today open // today candlestick is green
  // Yesterday close < Yesterday open // yesterday candlestick is red
  bool bullish_engulfing_pattern;
  void calculate_bullish_engulfing_pattern(std::vector<price> & prices)
  {
    if(index>=1)
    {
      bullish_engulfing_pattern = (close>prices[index-1].open)
                                &&(open<prices[index-1].close)
                                &&(close>open)
                                &&(prices[index-1].close<prices[index-1].open)
                                ;
    }
    else
    {
      bullish_engulfing_pattern = false;
    }
  }

  // Bearish Engulfing Pattern
  // Today open > Yesterday close
  // Today close < Yesterday open
  // Today close < Today open // today candlestick is red
  // Yesterday close > Yesterday open // yesterday candlestick is green
  bool bearish_engulfing_pattern;
  void calculate_bearish_engulfing_pattern(std::vector<price> & prices)
  {
    if(index>=1)
    {
      bearish_engulfing_pattern = (open>prices[index-1].close)
                                &&(close<prices[index-1].open)
                                &&(close<open)
                                &&(prices[index-1].close>prices[index-1].open)
                                ;
    }
    else
    {
      bearish_engulfing_pattern = false;
    }
  }

  static void initialize_engulfing_patterns(std::vector<price> & prices)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_bullish_engulfing_pattern(prices);
    }
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_bearish_engulfing_pattern(prices);
    }
  }

  // initialize all indicators 
  static void initialize_indicators(std::vector<price> & prices)
  {
    for(int i=0;i<prices.size();i++)
    {
      prices[i].calculate_ema_volume(prices,500);
    }
    initialize_CCI(prices);
    initialize_doji(prices);
    initialize_MACD(prices);
    initialize_GoldenCross(prices);
    initialize_engulfing_patterns(prices);
    initialize_RSI(prices);
    initialize_MFI(prices);
    initialize_Volume_spike(prices);
  }

};

struct Scanner
{
  std::set<std::string> buy;
  void scan(std::vector<std::vector<price> > & prices, std::vector<std::string> & symbols)
  {
    buy . clear();
    for(int i=0;i<prices.size();i++)
    {
      if(prices[i].size()>=3)
      {
        for(int d=0;d<3;d++)
        {
          if( prices[i][prices[i].size()-1-d].CCI_buy ||
              prices[i][prices[i].size()-1-d].RSI_buy ||
              prices[i][prices[i].size()-1-d].MFI_buy 
            )
          {
            buy.insert(symbols[i]);
          }
        }
      }
    }
    std::cout << "Buy candidates:" << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    for(std::set<std::string>::iterator it = buy.begin();it != buy.end();it++)
    {
      std::cout << *it << std::endl;
    }
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  }
};

bool comparator(price a,price b){return a.close < b.close;}
bool comparator_MACD(price a,price b){return a.MACD_dline < b.MACD_dline;}
bool comparator_low(price a,price b){return a.low < b.low;}
bool comparator_high(price a,price b){return a.high < b.high;}

bool comparator_volume(price a,price b){return a.volume < b.volume;}

struct Bin
{
  float price_min;
  float price_max;
  float sum;
  float sum_neg;
  float sum_pos;
  bool in(float price)
  {
    return price>=price_min&&price<price_max;
  }
  std::vector<price> collection;
  Bin(float _price_min,float _price_max)
  {
    price_min = _price_min;
    price_max = _price_max;
  }
  void push(price p)
  {
    if(in(p.close))
    {
      collection.push_back(p);
    }
  }
  void calc()
  {
    sum = 0;
    for(int i=0;i<collection.size();i++)
    {
      sum += collection[i].volume;
      if(collection[i].Volume_gain) sum_pos += collection[i].volume;
      if(collection[i].Volume_loss) sum_neg += collection[i].volume;
    }
  }
};

struct VolumeByPrice
{
  float max_sum;
  std::vector<Bin> bins;
  void create_bins(std::vector<price> & prices,int nbins,int min_index,int max_index)
  {
    float max_price = float(std::max_element(prices.begin()+min_index,prices.begin()+max_index,comparator)->close);
    float min_price = float(std::min_element(prices.begin()+min_index,prices.begin()+max_index,comparator)->close);
    float bin_size = (max_price-min_price)/nbins;
    for(int i=0;i<nbins;i++)
    {
      bins.push_back(Bin(min_price+bin_size*i,min_price+(i+1)*bin_size));
    }
    for(int i=min_index;i<=max_index;i++)
    {
      for(int k=0;k<nbins;k++)
      {
        bins[k].push(prices[i]);
      }
    }
    for(int i=0;i<nbins;i++)
    {
      bins[i].calc();
    }
    max_sum = 0;
    for(int i=0;i<nbins;i++)
    {
      if(bins[i].sum>max_sum)
      {
        max_sum = bins[i].sum;
      }
    }
    for(int i=0;i<nbins;i++)
    {
      bins[i].sum     /= max_sum+1;
      bins[i].sum_pos /= max_sum+1;
      bins[i].sum_neg /= max_sum+1;
    }
  }
};

void generate_synthetic(std::vector<price> & prices,float multiplier,int nyears)
{
  float value = (rand()%10000000)/10000.0;
  int date = 1;
  int nweeks = (int)(nyears*52.1429);
  for(int w=0;w<nweeks;w++)
  {
    for(int d=0;d<5;d++)
    {
      price p;
      std::stringstream date_ss;
      date_ss << date;
      p.index = prices.size();
      p.date = date_ss.str();
      p.open = value;
      p.close = value * ( 1.0 + multiplier*0.02 * (1-2*(rand()%10000)/10000.0) );
      p.high = max(p.open,p.close) * ( 1 + multiplier*0.02 * ((rand()%10000)/10000.0) );
      p.low = min(p.open,p.close) * ( 1 - multiplier*0.02 * ((rand()%10000)/10000.0) );
      p.volume = (int)(100*fabs(p.close-p.open));
      prices.push_back(p);
      if((rand()%10000)/10000.0 > 1.0/4.0)
      {
        value *= 1.0 + multiplier*0.01 * ((rand()%10000)/10000.0);
      }
      else
      {
        value /= 1.0 + multiplier*0.03 * ((rand()%10000)/10000.0);
      }
      date++;
    }
    date+=2;
  }
}

void read_data(std::string filename,std::vector<price> & prices)
{
  std::ifstream infile(filename.c_str());
  std::string line;
  int i=0;
  int dat=0;
  int day=0,month=0,year=0;
  while (std::getline(infile, line))
  {
    price D;
    std::stringstream iss(line);
    std::string token;
    // date
    iss >> token;
    boost::erase_all(token,"/");
    boost::erase_all(token,",");
    dat = atoi(token.c_str());
    year = dat%100;
    if(year > 50)year += 1900;else year += 2000;
    day = (dat/100)%100;
    month = (dat/10000)%100;
    //fprintf(stderr,"year:%d month:%d day:%d\n",year,month,day);
    D.date = token.c_str();
    // open
    iss >> token;
    boost::erase_all(token,",");
    D.open = atof(token.c_str());
    // high
    iss >> token;
    boost::erase_all(token,",");
    D.high = atof(token.c_str());
    // low
    iss >> token;
    boost::erase_all(token,",");
    D.low = atof(token.c_str());
    // close
    iss >> token;
    boost::erase_all(token,",");
    D.close = atof(token.c_str());
    D.volume = fabs(D.close - D.open)*100;
    prices.push_back(D);
  }
  infile.close();
  reverse(prices.begin(),prices.end());
  for(int i=0;i<prices.size();i++)
  {
    prices[i].index = i;
  }
}

void read_data_yahoo(std::string filename,std::vector<price> & prices)
{
  std::ifstream infile(filename.c_str());
  std::string line;
  int i=0;
  int dat=0;
  int day=0,month=0,year=0;
  while (std::getline(infile, line))
  {
    price D;
    boost::replace_all(line,","," ");
    std::stringstream iss(line);
    std::string token;
    // date
    iss >> token;
    D.date = token.c_str();
    // open
    iss >> token;
    boost::erase_all(token,",");
    D.open = atof(token.c_str());
    // high
    iss >> token;
    boost::erase_all(token,",");
    D.high = atof(token.c_str());
    // low
    iss >> token;
    boost::erase_all(token,",");
    D.low = atof(token.c_str());
    // close
    iss >> token;
    boost::erase_all(token,",");
    D.close = atof(token.c_str());
    // adj close
    iss >> token;
    boost::erase_all(token,",");
    //D.close = atof(token.c_str());
    // volume
    iss >> token;
    boost::erase_all(token,",");
    D.volume = atoi(token.c_str());
    if(D.close > 0 && D.low > 0 && D.high > 0 && D.open > 0)
    {
      prices.push_back(D);
    }
  }
  infile.close();
  for(int i=0;i<prices.size();i++)
  {
    prices[i].index = i;
  }
}

bool buy_only = false;
Scanner scanner;

std::vector<std::vector<price> > prices;
std::vector<std::string> symbols;

int width  = 1800;
int height = 1000;

int mouse_x = 0;
int mouse_y = 0;

int stock_index = 0;
float view_fraction = 1.0f;

int start_index = -1;
int   end_index = -1;
bool pick_start_index = false;
bool   pick_end_index = false;

void drawString (void * font, char const *s, float x, float y, float z)
{
     unsigned int i;
     glRasterPos3f(x, y, z);
     for (i = 0; i < strlen (s); i++)
     {
         glutBitmapCharacter (font, s[i]);
     }
}

void draw()
{
  glColor3f(1,1,1);
  drawString(GLUT_BITMAP_HELVETICA_18,symbols[stock_index].c_str(),-0.6,0.9,0);
  drawString(GLUT_BITMAP_HELVETICA_18,"Volume",-1,-0.10,0);
  drawString(GLUT_BITMAP_HELVETICA_18,"MACD",-1,-0.30,0);
  drawString(GLUT_BITMAP_HELVETICA_18,"RSI",-1,-0.50,0);
  drawString(GLUT_BITMAP_HELVETICA_18,"MFI",-1,-0.70,0);
  drawString(GLUT_BITMAP_HELVETICA_18,"CCI",-1,-0.90,0);
  int n = prices[stock_index].size()*view_fraction;
  int size = prices[stock_index].size()-1;
  float open_price = 0;
  float close_price = 0;
  float high_price = 0;
  float low_price = 0;
  std::string date = "";
  int price_index = (int)((size*(1.0f-view_fraction))+(((float)mouse_x/width)*(size*view_fraction))-0.5f) + 2;
  if(price_index>=0&&price_index<prices[stock_index].size())
  {
    open_price = prices[stock_index][price_index].open;
    close_price = prices[stock_index][price_index].close;
    high_price = prices[stock_index][price_index].high;
    low_price = prices[stock_index][price_index].low;
    date = prices[stock_index][price_index].date;
  }
  std::stringstream ss_date;
  ss_date << date << std::endl;
  std::stringstream ss_open_price;
  ss_open_price << "Open:" << open_price << std::endl;
  std::stringstream ss_close_price;
  ss_close_price << "Close:" << close_price << std::endl;
  std::stringstream ss_low_price;
  ss_low_price << "Low:" << low_price << std::endl;
  std::stringstream ss_high_price;
  ss_high_price << "High:" << high_price << std::endl;
  drawString(GLUT_BITMAP_HELVETICA_18,ss_date.str().c_str(),-1.0f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.20f,0);
  drawString(GLUT_BITMAP_HELVETICA_18,ss_open_price.str().c_str(),-1.0f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.15f,0);
  drawString(GLUT_BITMAP_HELVETICA_18,ss_close_price.str().c_str(),-1.0f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.10f,0);
  drawString(GLUT_BITMAP_HELVETICA_18,ss_high_price.str().c_str(),-1.0f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.05f,0);
  drawString(GLUT_BITMAP_HELVETICA_18,ss_low_price.str().c_str(),-1.0f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height+0.00f,0);
  VolumeByPrice vol_by_price;
  vol_by_price.create_bins(prices[stock_index],12,(int)(size*(1.0f-view_fraction)),size);
  float factor = 2.0f/n;
  float vfactor100 = 2.0f/100.0f;
  float vfactor400 = 2.0f/600.0f;
  float Bollinger_sigma = 1.0f;
  float vmin    = float(std::min_element(prices[stock_index].begin()+(int)(size*(1.0f-view_fraction)),prices[stock_index].end(),comparator_low )->low);
  float vmax    = float(std::max_element(prices[stock_index].begin()+(int)(size*(1.0f-view_fraction)),prices[stock_index].end(),comparator_high)->high);
  float MACD_min= float(std::min_element(prices[stock_index].begin()+(int)(size*(1.0f-view_fraction)),prices[stock_index].end(),comparator_MACD)->MACD_dline);
  float MACD_max= float(std::max_element(prices[stock_index].begin()+(int)(size*(1.0f-view_fraction)),prices[stock_index].end(),comparator_MACD)->MACD_dline);
  float MACD_cmp= max(fabs(MACD_min),fabs(MACD_max));
  //std::cout << MACD_min << "\t" << MACD_max << std::endl;
  float vfactor = 2.0f/(vmax-vmin);
  float vfactor_volume = 2.0f/float(std::max_element(prices[stock_index].begin()+(int)(size*(1.0f-view_fraction)),prices[stock_index].end(),comparator_volume)->volume);

  glBegin(GL_LINES);
  // mouse 
  glColor3f(1,1,1);
  glVertex3f( 100.0f+-1.0f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-100.0f+-1.0f+2.0f*mouse_x/width,1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-1.0f+2.0f*mouse_x/width, 100.0f+1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-1.0f+2.0f*mouse_x/width,-100.0f+1.0f-2.0f*mouse_y/height,0);
  glEnd();

  if(pick_start_index)
  {
    start_index = price_index;
    pick_start_index = false;
  }
  if(pick_end_index)
  {
    end_index = price_index;
    pick_end_index = false;
  }
  float chart_size = 0.05f;
  glBegin(GL_LINES);
  // MACD spiral axis 
  glColor3f(1.0,1.0,1.0);
  glVertex3f(-chart_size+ chart_size+-1.0f+2.0f*mouse_x/width,-chart_size            +1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+-chart_size+-1.0f+2.0f*mouse_x/width,-chart_size            +1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size            +-1.0f+2.0f*mouse_x/width,-chart_size+ chart_size+1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size            +-1.0f+2.0f*mouse_x/width,-chart_size+-chart_size+1.0f-2.0f*mouse_y/height,0);
  glEnd();
  glBegin(GL_LINES);
  // MACD current state
  glColor3f(1.0,1.0,1.0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].MACD_dline/MACD_cmp+ 0.01f+-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][price_index].MACD_dsignal)/MACD_cmp       +1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].MACD_dline/MACD_cmp+-0.01f+-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][price_index].MACD_dsignal)/MACD_cmp       +1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].MACD_dline/MACD_cmp       +-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][price_index].MACD_dsignal)/MACD_cmp+ 0.01f+1.0f-2.0f*mouse_y/height,0);
  glVertex3f(-chart_size+chart_size*prices[stock_index][price_index].MACD_dline/MACD_cmp       +-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][price_index].MACD_dsignal)/MACD_cmp+-0.01f+1.0f-2.0f*mouse_y/height,0);
  glEnd();
  glBegin(GL_LINES);
  // MACD spiral
  for(int i=1;i<n;i++)
  {
    glColor4f(1,1,1,.2f);
    if(size-i+1>start_index && size-i+1<end_index)
    {
      glColor4f(1,1,1,0.5f);
    }
    glVertex3f(-chart_size+chart_size*prices[stock_index][size-i+1].MACD_dline/MACD_cmp+-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][size-i+1].MACD_dsignal)/MACD_cmp+1.0f-2.0f*mouse_y/height,0);
    glVertex3f(-chart_size+chart_size*prices[stock_index][size-i  ].MACD_dline/MACD_cmp+-1.0f+2.0f*mouse_x/width,-chart_size+chart_size*(prices[stock_index][size-i  ].MACD_dsignal)/MACD_cmp+1.0f-2.0f*mouse_y/height,0);
  }
  glEnd();

  glBegin(GL_QUADS);
  for(int i=0;i<vol_by_price.bins.size();i++)
  {
    glColor3f(1,0,0);
    glVertex3f(-1.0f                                   , 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_min-vmin) ,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum_neg, 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_min-vmin) ,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum_neg, 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_max-vmin) ,0);
    glVertex3f(-1.0f                                   , 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_max-vmin) ,0);
    glColor3f(0,1,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum_neg, 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_min-vmin) ,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum    , 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_min-vmin) ,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum    , 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_max-vmin) ,0);
    glVertex3f(-1.0f+0.25f*vol_by_price.bins[i].sum_neg, 0.0f+0.5f*vfactor*(vol_by_price.bins[i].price_max-vmin) ,0);
  }
  glEnd();

  glBegin(GL_LINES);
  for(int i=1;i<n;i++)
  {
    glColor3f(1,1,1);

    // volume
    glVertex3f(1.0f- i   *factor,-0.2f+0.1f*vfactor_volume*prices[stock_index][size-i+1].volume,0);
    glVertex3f(1.0f-(i+1)*factor,-0.2f+0.1f*vfactor_volume*prices[stock_index][size-i  ].volume,0);
    glVertex3f(1.0f- i   *factor,-0.2f+0.1f*vfactor_volume*prices[stock_index][size-i+1].EMAV  ,0);
    glVertex3f(1.0f-(i+1)*factor,-0.2f+0.1f*vfactor_volume*prices[stock_index][size-i  ].EMAV  ,0);

    // MACD
    glVertex3f(1.0f- i   *factor,-0.4f+0.1f*vfactor400*(300+100.0f*prices[stock_index][size-i+1].MACD_dline/MACD_cmp),0);
    glVertex3f(1.0f-(i+1)*factor,-0.4f+0.1f*vfactor400*(300+100.0f*prices[stock_index][size-i  ].MACD_dline/MACD_cmp),0);
    glColor3f(1,0,0);
    glVertex3f(1.0f- i   *factor,-0.4f+0.1f*vfactor400*(300+100.0f*prices[stock_index][size-i+1].MACD_dsignal/MACD_cmp),0);
    glVertex3f(1.0f-(i+1)*factor,-0.4f+0.1f*vfactor400*(300+100.0f*prices[stock_index][size-i  ].MACD_dsignal/MACD_cmp),0);
    glColor3f(1,1,1);
    glVertex3f(1.0f- i   *factor,-0.4f+0.1f*vfactor400*(300+0),0);
    glVertex3f(1.0f-(i+1)*factor,-0.4f+0.1f*vfactor400*(300+0),0);

    // RSI
    glVertex3f(1.0f- i   *factor,-0.6f+0.1f*vfactor100*prices[stock_index][size-i+1].RSI,0);
    glVertex3f(1.0f-(i+1)*factor,-0.6f+0.1f*vfactor100*prices[stock_index][size-i  ].RSI,0);
    glVertex3f(1.0f- i   *factor,-0.6f+0.1f*vfactor100*30,0);
    glVertex3f(1.0f-(i+1)*factor,-0.6f+0.1f*vfactor100*30,0);
    glVertex3f(1.0f- i   *factor,-0.6f+0.1f*vfactor100*50,0);
    glVertex3f(1.0f-(i+1)*factor,-0.6f+0.1f*vfactor100*50,0);
    glVertex3f(1.0f- i   *factor,-0.6f+0.1f*vfactor100*70,0);
    glVertex3f(1.0f-(i+1)*factor,-0.6f+0.1f*vfactor100*70,0);

    // MFI
    glVertex3f(1.0f- i   *factor,-0.8f+0.1f*vfactor100*prices[stock_index][size-i+1].MFI,0);
    glVertex3f(1.0f-(i+1)*factor,-0.8f+0.1f*vfactor100*prices[stock_index][size-i  ].MFI,0);
    glVertex3f(1.0f- i   *factor,-0.8f+0.1f*vfactor100*30,0);
    glVertex3f(1.0f-(i+1)*factor,-0.8f+0.1f*vfactor100*30,0);
    glVertex3f(1.0f- i   *factor,-0.8f+0.1f*vfactor100*50,0);
    glVertex3f(1.0f-(i+1)*factor,-0.8f+0.1f*vfactor100*50,0);
    glVertex3f(1.0f- i   *factor,-0.8f+0.1f*vfactor100*70,0);
    glVertex3f(1.0f-(i+1)*factor,-0.8f+0.1f*vfactor100*70,0);

    // CCI
    glVertex3f(1.0f- i   *factor,-1.0f+0.1f*vfactor400*(300+prices[stock_index][size-i+1].CCI),0);
    glVertex3f(1.0f-(i+1)*factor,-1.0f+0.1f*vfactor400*(300+prices[stock_index][size-i  ].CCI),0);
    glVertex3f(1.0f- i   *factor,-1.0f+0.1f*vfactor400*(300+ 100),0);
    glVertex3f(1.0f-(i+1)*factor,-1.0f+0.1f*vfactor400*(300+ 100),0);
    glVertex3f(1.0f- i   *factor,-1.0f+0.1f*vfactor400*(300+   0),0);
    glVertex3f(1.0f-(i+1)*factor,-1.0f+0.1f*vfactor400*(300+   0),0);
    glVertex3f(1.0f- i   *factor,-1.0f+0.1f*vfactor400*(300+-100),0);
    glVertex3f(1.0f-(i+1)*factor,-1.0f+0.1f*vfactor400*(300+-100),0);
  }
  glEnd();

  glBegin(GL_QUADS);
  for(int i=1;i<n;i++)
  {
    if(prices[stock_index][size-i+1].close>prices[stock_index][size-i].close)
    {
      glColor3f(0,1,0);
    }
    else
    {
      glColor3f(1,0,0);
    }
    // volume
    glVertex3f(1.0f-(i-0.45f)*factor,-0.25f+0.125f*vfactor_volume*prices[stock_index][size-i+1].volume,0);
    glVertex3f(1.0f-(i+0.45f)*factor,-0.25f+0.125f*vfactor_volume*prices[stock_index][size-i+1].volume,0);
    glVertex3f(1.0f-(i+0.45f)*factor,-0.25f,0);
    glVertex3f(1.0f-(i-0.45f)*factor,-0.25f,0);
  }
  glEnd();

  glBegin(GL_QUADS);
  {
    glColor3f(0,0,0);
    glVertex3f(-1,-1,0);
    glVertex3f(1,-1,0);
    glVertex3f(1,0,0);
    glVertex3f(-1,0,0);
  }
  glEnd();
    
  glBegin(GL_LINES);
  for(int i=1;i<n;i++)
  {
    glColor3f(1,1,1);
    // price
    glVertex3f(1.0f- i   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].close-vmin) ,0);
    glVertex3f(1.0f-(i+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].close-vmin) ,0);

    glVertex3f(1.0f- i   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_12-vmin) ,0);
    glVertex3f(1.0f-(i+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_12-vmin) ,0);
    for(int b=1;b<=3;b++)
    {
      glColor3f(1.0f/(1+b),1.0f/(1+b),1.0f/(1+b));
      glVertex3f(1.0f- i   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_12+b*Bollinger_sigma*prices[stock_index][size-i+1].ems_12-vmin) ,0);
      glVertex3f(1.0f-(i+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_12+b*Bollinger_sigma*prices[stock_index][size-i  ].ems_12-vmin) ,0);
      glVertex3f(1.0f- i   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_12-b*Bollinger_sigma*prices[stock_index][size-i+1].ems_12-vmin) ,0);
      glVertex3f(1.0f-(i+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_12-b*Bollinger_sigma*prices[stock_index][size-i  ].ems_12-vmin) ,0);
    }
    
    //glVertex3f(1.0f- i   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_26-vmin) ,0);
    //glVertex3f(1.0f-(i+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_26-vmin) ,0);
    //for(int b=1;b<=3;b++)
    //{
    //  glColor3f(1.0f/(1+b),1.0f/(1+b),1.0f/(1+b));
    //  glVertex3f(1.0f- i   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_26+b*Bollinger_sigma*prices[stock_index][size-i+1].ems_26-vmin) ,0);
    //  glVertex3f(1.0f-(i+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_26+b*Bollinger_sigma*prices[stock_index][size-i  ].ems_26-vmin) ,0);
    //  glVertex3f(1.0f- i   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_26-b*Bollinger_sigma*prices[stock_index][size-i+1].ems_26-vmin) ,0);
    //  glVertex3f(1.0f-(i+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_26-b*Bollinger_sigma*prices[stock_index][size-i  ].ems_26-vmin) ,0);
    //}
    
    //glVertex3f(1.0f- i   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_50-vmin) ,0);
    //glVertex3f(1.0f-(i+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_50-vmin) ,0);
    //for(int b=1;b<=3;b++)
    //{
    //  glColor3f(1.0f/(1+b),1.0f/(1+b),1.0f/(1+b));
    //  glVertex3f(1.0f- i   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_50+b*Bollinger_sigma*prices[stock_index][size-i+1].ems_50-vmin) ,0);
    //  glVertex3f(1.0f-(i+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_50+b*Bollinger_sigma*prices[stock_index][size-i  ].ems_50-vmin) ,0);
    //  glVertex3f(1.0f- i   *factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].ema_50-b*Bollinger_sigma*prices[stock_index][size-i+1].ems_50-vmin) ,0);
    //  glVertex3f(1.0f-(i+1)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i  ].ema_50-b*Bollinger_sigma*prices[stock_index][size-i  ].ems_50-vmin) ,0);
    //}
  } 
  glEnd();

  glBegin(GL_QUADS);
  for(int i=1;i<n;i++)
  {
    if(size-i+1 == price_index)
    {
      if(prices[stock_index][size-i+1].close>prices[stock_index][size-i+1].open)
      {
        glColor3f(0,1,0);
      }
      else
      {
        glColor3f(1,0,0);
      }
    }
    else
    {
      if(prices[stock_index][size-i+1].close>prices[stock_index][size-i+1].open)
      {
        glColor3f(0,1,1);
      }
      else
      {
        glColor3f(1,0.5f,0);
      }
    }
    // price
    glVertex3f(1.0f-(i-0.45f)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].open -vmin) ,0);
    glVertex3f(1.0f-(i+0.45f)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].open -vmin) ,0);
    glVertex3f(1.0f-(i+0.45f)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].close-vmin) ,0);
    glVertex3f(1.0f-(i-0.45f)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].close-vmin) ,0);

    glVertex3f(1.0f-(i-0.05f)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].low  -vmin) ,0);
    glVertex3f(1.0f-(i+0.05f)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].low  -vmin) ,0);
    glVertex3f(1.0f-(i+0.05f)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].high -vmin) ,0);
    glVertex3f(1.0f-(i-0.05f)*factor, 0.0f+0.5f*vfactor       *(prices[stock_index][size-i+1].high -vmin) ,0);
  }
  glEnd();

}

void display()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  draw();
  glutSwapBuffers();
}

void idle()
{
  usleep(10000);
  glutPostRedisplay();
}

void init()
{
  /* Use depth buffering for hidden surface elimination. */
  glEnable(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 10.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 0.0, 1.8,  /* eye is at (0,0,5) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 1.0, 0.);      /* up is in positive Y direction */

  /* Adjust cube position to be asthetic angle. */
  glTranslatef(0.0, 0.0, -1.0);
  glRotatef(0, 1.0, 0.0, 0.0);
  glRotatef(0, 0.0, 0.0, 1.0);
  glEnable (GL_BLEND); 
  //glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE);
  glBlendEquation(GL_FUNC_ADD);
}

void keyboard(unsigned char key,int x,int y)
{
  switch(key)
  {
    case 'a':
      if(buy_only==false)
      {
        stock_index--;
        if(stock_index<0)stock_index=prices.size()-1;
      }
      else
      {
        if(scanner.buy.size()==0)
        {
          buy_only=false;
          stock_index--;
          if(stock_index<0)stock_index=prices.size()-1;
        }
        else
        {
          while(true)
          {
            stock_index--;
            if(stock_index<0)stock_index=prices.size()-1;
            if(scanner.buy.find(symbols[stock_index]) != scanner.buy.end())break;
          }
        }
      }
      std::cout << symbols[stock_index] << std::endl;
      break;
    case 'd':
      if(buy_only==false)
      {
        stock_index++;
        if(stock_index>=prices.size())stock_index=0;
      }
      else
      {
        if(scanner.buy.size()==0)
        {
          buy_only=false;
          stock_index++;
          if(stock_index>=prices.size())stock_index=0;
        }
        else
        {
          while(true)
          {
            stock_index++;
            if(stock_index>=prices.size())stock_index=0;
            if(scanner.buy.find(symbols[stock_index]) != scanner.buy.end())break;
          }
        }
      }
      std::cout << symbols[stock_index] << std::endl;
      break;
    case 'w':view_fraction *= 1.1f;if(view_fraction>1.0f)view_fraction=1.0f;break;
    case 's':view_fraction /= 1.1f;break;
    case 'c':pick_start_index = true;break;
    case 'v':pick_end_index = true;break;
    case 'b':start_index=-1;end_index=-1;break;
    case 'z':
      buy_only = !buy_only;
      if(buy_only) 
      {
        if(scanner.buy.size()==0)
        {
          buy_only=false;
          stock_index++;
          if(stock_index>=prices.size())stock_index=0;
        }
        else
        {
          while(true)
          {
            stock_index++;
            if(stock_index>=prices.size())stock_index=0;
            if(scanner.buy.find(symbols[stock_index]) != scanner.buy.end())break;
          }
        }
      }
      break;
    case 27:exit(1);break;
    default:break;
  }
}

void passive_mouse(int x,int y)
{
  mouse_x = x;
  mouse_y = y;
}

void active_mouse(int x,int y)
{
  mouse_x = x;
  mouse_y = y;
}

int main(int argc,char ** argv)
{
  int seed;
  seed = time(0);
  srand(seed);
  
  // list all files in current directory.
  boost::filesystem::path p ("data");
  boost::filesystem::directory_iterator end_itr;
  // cycle through the directory
  int ind = 0;
  for (boost::filesystem::directory_iterator itr(p); itr != end_itr; ++itr,ind++)
  {
    // If it's not a directory, list it. If you want to list directories too, just remove this check.
    if (boost::filesystem::is_regular_file(itr->path())) {
      prices.push_back(std::vector<price>());
      // assign current file name to current_file and echo it out to the console.
      std::string current_file = itr->path().string();
      symbols.push_back(current_file);
      fprintf(stderr,"file:%s\n",current_file.c_str());
      read_data_yahoo(current_file,prices[ind]);
    }
  }


  //for(int i=0;i<prices[0].size();i++)
  //{
  //  std::cout << prices[0][i].close << std::endl;
  //}
  //std::cout << "0\n0\n0\n0\n0" << std::endl;
  for(int i=0;i<prices.size();i++)
  {
    price::initialize_indicators(prices[i]);
  }
  //for(int i=0;i<prices.size();i++)
  //{
  //  std::cout << prices[i].close << " " << prices[i].RSI << " " << prices[i].average_gain << " " << prices[i].average_loss << std::endl;
  //  std::cout << prices[i].RSI << std::endl;
  //}
  //std::cout << "0\n0\n0\n0\n0" << std::endl;
  //for(int i=0;i<prices.size();i++)
  //{
  //  std::cout << ((prices[i].MFI_buy)?0.0f:prices[i].close) << std::endl;
  //}
  //std::cout << "0\n0\n0\n0\n0" << std::endl;
  //for(int i=0;i<prices.size();i++)
  //{
  //  std::cout << ((prices[i].MFI_sell)?0.0f:prices[i].close) << std::endl;
  //}
  //std::cout << "0\n0\n0\n0\n0" << std::endl;
  //for(int i=0;i<prices.size();i++)
  //{
  //  std::cout << ((prices[i].Volume_spike)?0.0f:prices[i].close) << std::endl;
  //}
  //std::cout << "0\n0\n0\n0\n0" << std::endl;
  //for(int i=0;i<prices[0].size();i++)
  //{
  //  std::cout << prices[0][i].SMAV << std::endl;
  //}
  //std::cout << "0\n0\n0\n0\n0" << std::endl;

  scanner.scan(prices,symbols);

  glutInit(&argc, argv);
  glutInitWindowSize(width,height);
  glutInitWindowPosition(0,0);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutCreateWindow("stock bot");
  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutPassiveMotionFunc(passive_mouse);
  glutMotionFunc(active_mouse);
  glutIdleFunc(idle);
  init();
  glutMainLoop();
  return 0;
}

