public class SleepThreadTest {
	public static void main(String[] args) {
		System.out.println("main thread started");
		SleepThread thread1 = new SleepThread("thread1");
		SleepThread thread2 = new SleepThread("thread2");
		SleepThread thread3 = new SleepThread("thread3");
		
		thread1.start();
		thread2.start();
		thread3.start();

		System.out.println("main thread started 3 sub-threads and self-ended");
	}
}

class SleepThread extends Thread {
	int sleeptime_ms;

	public SleepThread(String name) {
		super(name);
		sleeptime_ms = (int)(Math.random()*100) + 1;
	}

	public void run() {
		System.out.println(getName() + " started and will sleep " + sleeptime_ms + " ms");
		try{
			Thread.sleep(sleeptime_ms);
		} catch (Exception e) {;}
		System.out.println(getName() + " ended");		
	}
}