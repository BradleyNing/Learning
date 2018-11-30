public class SleepClassRunnableTest {
	public static void main(String[] args) {
		System.out.println("main thread started");
		SleepClassRunnable thread1 = new SleepClassRunnable();
		//SleepClassRunnable thread2 = new SleepClassRunnable();
		//SleepClassRunnable thread3 = new SleepClassRunnable();
		
		//new Thread(thread1, "thread1").start();
		//new Thread(thread2, "thread2").start();		
		//new Thread(thread1, "thread2").start();
		//new Thread(thread3, "thread3").start();
		//new Thread(thread1, "thread3").start();
		new Thread(thread1).start();
		new Thread(thread1).start();
		new Thread(thread1).start();

		System.out.println("main thread started 3 sub-threads and self-ended");
	}
}

class SleepClassRunnable implements Runnable {
	int sleeptime_ms;

	public SleepClassRunnable() {
		sleeptime_ms = (int)(Math.random()*100) + 1;
	}

	public void run() {
		System.out.println(Thread.currentThread().getName() + " started and will sleep " + sleeptime_ms + " ms");
		try{
			Thread.sleep(sleeptime_ms);
		} catch (Exception e) {;}
		System.out.println(Thread.currentThread().getName() + " ended");		
	}
}